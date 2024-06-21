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
import math
import copy
from functools import partial
from typing import Optional, Union

import torch
from torch import nn

from fourm.utils.timm.registry import register_model
from huggingface_hub import PyTorchModelHubMixin

from .encoder_embeddings import ImageEncoderEmbedding
from .fm_utils import Block, LayerNorm
from fourm.data.modality_info import MODALITY_INFO


__all__ = [
    # GELU models
    'fm_vit_tiny_6e_gelu',
    'fm_vit_small_8e_gelu',
    'fm_vit_base_12e_gelu',
    'fm_vit_large_24e_gelu',
    'fm_vit_xlarge_24e_gelu',
    # SwiGLU models
    'fm_vit_tiny_6e_swiglu_nobias',
    'fm_vit_small_8e_swiglu_nobias',
    'fm_vit_base_12e_swiglu_nobias',
    'fm_vit_large_24e_swiglu_nobias',
    'fm_vit_xlarge_24e_swiglu_nobias',
    # SwiGLU + QKNorm models
    'fm_vit_base_12e_swiglu_qknorm_nobias',
    'fm_vit_large_24e_swiglu_qknorm_nobias',
    'fm_vit_xlarge_24e_swiglu_qknorm_nobias',
]

class FourMViT(nn.Module):
    """Modified 4M model, adapted to behave as a simple RGB-only ViT.

    Args:
        img_size (int): Input image size.
        patch_size (int): Patch size.
        in_chans (int): Number of input image channels.
        dim (int): Patch embedding dimension.
        encoder_depth (int): Depth of ViT / number of encoder blocks.
        num_heads (int): Number of attention heads in each ViT block.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        proj_bias (bool): If True, adds a bias to the attention out proj layer.
        mlp_bias (bool): If True, adds a learnable bias for the feedforward.
        drop_path_rate (float): Stochastic depth rate.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate.
        act_layer (nn.Module): Activation layer.
        norm_layer (nn.Module): Normalization layer.
        gated_mlp (bool): If True, makes the feedforward gated (e.g., for SwiGLU)
        qk_norm (bool): If True, normalizes the query and keys (as in ViT-22B)
        use_act_checkpoint (bool): If True, use activation checkpointing.
        encoder_norm (bool): If True, adds a norm layer after the last encoder block.
        output_head (Optional[nn.Module]): Optional output head after the encoder
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        dim=768,
        encoder_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_bias: bool = True,
        drop_path_rate: float =0.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float =0.0,
        act_layer: torch.Tensor =nn.GELU,
        norm_layer: Union[partial, nn.Module] = partial(LayerNorm, eps=1e-6),
        gated_mlp: bool = False, # Make the feedforward gated for e.g. SwiGLU
        qk_norm: bool = False,
        encoder_norm = True,
        output_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.init_std = 0.02
        rgb_embedding = ImageEncoderEmbedding(num_channels=in_chans, patch_size=patch_size,
                                              dim_tokens=dim, sincos_pos_emb=True, image_size=img_size)
        self.num_patches = rgb_embedding.num_patches
        self.encoder_embeddings = nn.ModuleDict({f"rgb@{img_size}": rgb_embedding})

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, encoder_depth)]

        self.encoder = nn.ModuleList([
            Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias, mlp_bias=mlp_bias,
                 drop_path=dpr[i], drop=drop_rate, attn_drop=attn_drop_rate, act_layer=act_layer, norm_layer=norm_layer, 
                 gated_mlp=gated_mlp, qk_norm=qk_norm)
            for i in range(encoder_depth)
        ])

        self.encoder_norm = norm_layer(dim) if encoder_norm else nn.Identity()

        # Weight init
        self.init_weights()

        # Classification head is initialized after init_weights() to allow for special init scale
        if output_head is not None:
            self.output_head = output_head
            if hasattr(self.output_head, 'init'):
                self.output_head.init(dim)
        else:
            self.output_head = nn.Identity()

    def init_weights(self):
        """Weight initialization following MAE's initialization scheme"""

        for name, m in self.named_modules():
            # Skipping tokenizers to avoid reinitializing them
            if "tokenizer" in name:
                continue
            # Linear
            elif isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif 'kv' in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm) or isinstance(m, LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

            # Embedding
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=self.init_std)
            # Conv2d
            elif isinstance(m, nn.Conv2d):
                if '.proj' in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def get_num_layers_encoder(self):
        return len(self.encoder)
    
    def get_num_layers(self):
        return self.get_num_layers_encoder()
    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = set()

        for mod, emb_module in self.encoder_embeddings.items():
            if hasattr(emb_module, 'no_weight_decay'):
                to_skip = emb_module.no_weight_decay()
                to_skip = set([f'encoder_embeddings.{mod}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        return no_wd_set
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor. Shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor. Shape (B, num_classes).
        """
        rgb_dict = {'tensor': x}
        rgb_dict = self.encoder_embeddings[f'rgb@{self.img_size}'](rgb_dict)

        # Add embeddings to patchified RGB image 
        x = rgb_dict['x'] + rgb_dict['emb'] # Shape: (B, N, D) with N = num_patches

        for blk in self.encoder:
            x = blk(x)

        x = self.encoder_norm(x) # Shape: (B, N, D) 

        out = self.output_head(x)

        return out
    
    
    def freeze_encoder(self, freeze_embeddings=True):
        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder_norm.parameters():
            param.requires_grad = False

        if freeze_embeddings:
            for param in self.encoder_embeddings.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self, unfreeze_embeddings=True):
        for param in self.encoder.parameters():
            param.requires_grad = True

        for param in self.encoder_norm.parameters():
            param.requires_grad = True

        if unfreeze_embeddings:
            for param in self.encoder_embeddings.parameters():
                param.requires_grad = True


################################################

# Wrapper for easy loading with Huggingface Hub

class FMViT(FourMViT, PyTorchModelHubMixin):
    """Wrapper around FourMViT for easy loading with Huggingface Hub.

    Args:
        config (dict): Dictionary containing the model and modality configuration, 
            used for loading from Huggingface Hub.
        output_head (nn.Module): Optional output head.
    """
    def __init__(self, config: dict, output_head: Optional[nn.Module] = None):

        config = copy.deepcopy(config)

        

        config['norm_layer'] = partial(LayerNorm, eps=1e-6, bias=config['norm_bias'])
        config['act_layer'] = getattr(torch.nn, config['act_layer'])

        img_size = config['image_size']
        config['img_size'] = img_size
        config['patch_size'] = MODALITY_INFO[f'rgb@{img_size}'].get('patch_size', config['patch_size'])
        config['in_chans'] = MODALITY_INFO[f'rgb@{img_size}'].get('num_channels', 3)

        for key in ['image_size', 'norm_bias', 'domains_in', 'domains_out', 'decoder_depth', 'share_modality_embeddings']:
            if key in config:
                del config[key]

        super().__init__(
            output_head=output_head,
            **config
        )   


################################################

# Model definitions
                
# GELU variants
@register_model
def fm_vit_tiny_6e_gelu(**kwargs):
    model = FourMViT(
        encoder_depth=6,
        dim=384,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


@register_model
def fm_vit_small_8e_gelu(**kwargs):
    model = FourMViT(
        encoder_depth=8,
        dim=512,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


@register_model
def fm_vit_base_12e_gelu(**kwargs):
    model = FourMViT(
        encoder_depth=12,
        dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


@register_model
def fm_vit_large_24e_gelu(**kwargs):
    model = FourMViT(
        encoder_depth=24,
        dim=1024,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

@register_model
def fm_vit_xlarge_24e_gelu(**kwargs):
    model = FourMViT(
        encoder_depth=24,
        dim=2048,
        num_heads=32,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


# SwiGLU variants
@register_model
def fm_vit_tiny_6e_swiglu_nobias(**kwargs):
    model = FourMViT(
        encoder_depth=6,
        dim=384,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        **kwargs
    )
    return model


@register_model
def fm_vit_small_8e_swiglu_nobias(**kwargs):
    model = FourMViT(
        encoder_depth=8,
        dim=512,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        **kwargs
    )
    return model


@register_model
def fm_vit_base_12e_swiglu_nobias(**kwargs):
    model = FourMViT(
        encoder_depth=12,
        dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        **kwargs
    )
    return model


@register_model
def fm_vit_large_24e_swiglu_nobias(**kwargs):
    model = FourMViT(
        encoder_depth=24,
        dim=1024,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        **kwargs
    )
    return model

@register_model
def fm_vit_xlarge_24e_swiglu_nobias(**kwargs):
    model = FourMViT(
        encoder_depth=24,
        dim=2048,
        num_heads=32,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        **kwargs
    )
    return model

# SwiGLU + QKNorm variants

@register_model
def fm_vit_base_12e_swiglu_qknorm_nobias(**kwargs):
    model = FourMViT(
        encoder_depth=12,
        dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        qk_norm=True,
        **kwargs
    )
    return model


@register_model
def fm_vit_large_24e_swiglu_qknorm_nobias(**kwargs):
    model = FourMViT(
        encoder_depth=24,
        dim=1024,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        qk_norm=True,
        **kwargs
    )
    return model

@register_model
def fm_vit_xlarge_24e_swiglu_qknorm_nobias(**kwargs):
    model = FourMViT(
        encoder_depth=24,
        dim=2048,
        num_heads=32,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        qk_norm=True,
        **kwargs
    )
    return model