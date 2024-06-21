# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Set, Optional, Type

import torch
import torch.nn as nn


SELF_ATTENTION_MODULES = {'Attention', 'NormAttention'}
CROSS_ATTENTION_MODULES = {'CrossAttention', 'NormCrossAttention'}
ATTENTION_MODULES = SELF_ATTENTION_MODULES | CROSS_ATTENTION_MODULES
MLP_MODULES = {'Mlp', 'GatedMlp', 'SwiGLUFFNFused'} # SwiGLUFFNFused is from DINOv2
TRANSFORMER_MODULES = ATTENTION_MODULES | MLP_MODULES


def get_LoRA_module_names(id: str) -> Set[str]:
    """ Returns a list of module names that are LoRA-adapted for the given id. """
    id = id.lower()
    if id in ['selfattn', 'selfattention', 'self_attn', 'self_attention']:
        return SELF_ATTENTION_MODULES
    elif id in ['crossattn', 'crossattention', 'cross_attn', 'cross_attention']:
        return CROSS_ATTENTION_MODULES
    elif id in ['attn', 'attention']:
        return ATTENTION_MODULES
    elif id in ['mlp']:
        return MLP_MODULES
    elif id in ['all', 'transformer']:
        return TRANSFORMER_MODULES
    else:
        raise ValueError(f'Unknown LoRA module id {id}.')


class LoRAWrapper(nn.Module):
    """Low-Rank Adaptation Wrapper for linear layers.
    See https://arxiv.org/abs/2106.09685
    
    Args:
        linear: nn.Linear layer to wrap
        rank: Rank of adaptation matrix B@A
        scale: x = W_0@x + scale * B@A@x
        num_packed_linear: Set to > 1 when wrapping e.g. packed kv, or qkv attention weights.
            Weights will be initialized as if num_packed_linear = 1, but the LoRA bottleneck will
            be num_packed_linear times larger.
    """
    def __init__(self, linear: nn.Module, rank: int = 4, scale: float = 1.0, num_packed_linear: int = 1):
        super().__init__()
        self.rank = rank
        self.scale = scale
        self.in_features, self.out_features = linear.in_features, linear.out_features
        assert num_packed_linear * rank <= min(self.in_features, self.out_features), \
            f'LoRA rank {num_packed_linear} * {rank} must be less or equal than {min(self.in_features, self.out_features)}'
        
        self.linear = linear
        self.lora_down = nn.Linear(self.in_features, num_packed_linear*rank, bias=False)
        self.lora_up = nn.Linear(num_packed_linear*rank, self.out_features, bias=False)

        nn.init.normal_(self.lora_down.weight, std=1/rank)
        nn.init.zeros_(self.lora_up.weight)
        
    def fuse_LoRA_into_linear(self) -> nn.Linear:
        """ Returns a single nn.Linear layer with the LoRA matrix fused into the original one. """
        fused_linear = nn.Linear(self.in_features, self.out_features, bias=self.linear.bias is not None)
        fused_linear.weight.data = self.linear.weight + self.scale * (self.lora_up.weight @ self.lora_down.weight)
        if self.linear.bias is not None:
            fused_linear.bias.data = self.linear.bias
        return fused_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ LoRA adapted linear layer forward pass. """
        return self.linear(x) + self.lora_up(self.lora_down(x)) * self.scale
    

def _find_modules(
    model,
    ancestor_class: Optional[Set[str]] = None,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [LoRAWrapper],
):
    """
    Find all modules of a certain class (or union of classes) that are direct or
    indirect descendants of other modules of a certain class (or union of classes).

    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    
    Adapted from https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
    """
    # Get the targets we should replace all linears under
    if ancestor_class is not None:
        ancestors = (
            module
            for module in model.modules()
            if module.__class__.__name__ in ancestor_class
        )
    else:
        # this, incase you want to naively iterate over all modules.
        ancestors = [module for module in model.modules()]

    # For each target find every linear_class module that isn't a child of a LoRA layer
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a LoRA layer
                if exclude_children_of and any(
                    [isinstance(parent, _class) for _class in exclude_children_of]
                ):
                    continue
                # Otherwise, yield it
                yield parent, name, module
                

def inject_trainable_LoRA(
    model: nn.Module, 
    rank: int = 4, 
    scale: float = 1.0,
    target_replace_modules: Set[str] = ATTENTION_MODULES
) -> None:
    """Replaces all linear layers of the specified modules with LoRA-adapted linear layers.
    Modifies the model in-place.
    
    Args:
        model: nn.Module to modify
        rank: Rank of adaptation matrix B@A
        scale: x = W_0@x + scale * B@A@x
        target_replace_modules: Set of module names to replace linear layers in.
    """
    for _module, name, _child_module in _find_modules(
        model, target_replace_modules, search_class=[nn.Linear]
    ):
        if sorted(name) == sorted('qkv'):
            num_packed_linear = 3
        elif sorted(name) in [sorted('kv'), sorted('qk'), sorted('qv')]:
            num_packed_linear = 2
        else:
            num_packed_linear = 1
        
        _module._modules[name] = LoRAWrapper(_child_module, rank=rank, scale=scale, num_packed_linear=num_packed_linear)
        

def fuse_LoRA_into_linear(
    model: nn.Module,
    target_replace_modules: Set[str] = ATTENTION_MODULES
) -> None:
    """Fuses all LoRA-adapted linear layers back into single linear layers.
    Modifies the model in-place.

    Args:
        model: nn.Module to modify
        target_replace_modules: Set of module names to replace linear layers in.
    """
    for _module, name, _child_module in _find_modules(
        model, target_replace_modules, search_class=[LoRAWrapper]
    ):
        _module._modules[name] = _module._modules[name].fuse_LoRA_into_linear()


def unfreeze_all_LoRA_layers(model: nn.Module) -> None:
    """ Unfreezes all LoRA-adapted linear layers. """
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
