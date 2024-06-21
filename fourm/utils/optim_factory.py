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
# --------------------------------------------------------
# Based on BEiT, timm, DINO, DeiT code base
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------
import json

import torch
from torch import optim as optim


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed", "global_tokens"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("input_adapters") or var_name.startswith("encoder_embeddings"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks") or (var_name.startswith("encoder.") and not var_name.startswith("encoder_norm")):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


def get_num_layer_for_beit(var_name, num_max_layer):
    if "embed" in var_name:
        return 0
    elif var_name in (
        "cls_token", "mask_token", "pos_embed", "language_pos_embed", 
        "word_embeddings.weight", "vision_cls_token", "vision_pos_embed"
    ):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif "layers." in var_name:
        layer_id = int(var_name.split('layers.')[1].split('.')[0])
        return layer_id + 1
    else:
        return num_max_layer - 1


def get_num_layer_for_fm(var_name, num_enc_layers, num_dec_layers, last_layer_mod_emb=False):
    """Layers go from 0 to (num_enc + num_dec + 1)
    where 0 is the encoder embedding and (num_enc + num_dec + 1) is the projection following the decoder
    """
    if var_name.startswith("encoder_embeddings"):
        return 0
    elif var_name.startswith("encoder."):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    elif var_name in ("encoder_norm", "decoder_proj_context", "mask_token"):
        return num_enc_layers
    elif not last_layer_mod_emb and var_name.startswith("decoder_embeddings.") and "mod_emb" in var_name:
        return num_enc_layers
    elif var_name.startswith("decoder."):
        layer_id = int(var_name.split('.')[1])
        return num_enc_layers + layer_id + 1
    else:
        return num_enc_layers + num_dec_layers + 1


class LayerDecayValueAssigner(object):
    def __init__(self, values, is_beit3=False):
        self.values = values
        self.is_beit3 = is_beit3

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        if self.is_beit3:
            return get_num_layer_for_beit(var_name, len(self.values))
        else:
            return get_num_layer_for_vit(var_name, len(self.values))

class LayerDecayValueAssignerForFourM(object):
    def __init__(self, values, num_enc_layers, num_dec_layers, last_layer_mod_emb=False):
        self.values = values
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.last_layer_mod_emb = last_layer_mod_emb
        assert len(values) == num_enc_layers + num_dec_layers + 2

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_fm(var_name, num_enc_layers=self.num_enc_layers, num_dec_layers=self.num_dec_layers, last_layer_mod_emb=self.last_layer_mod_emb)


def get_parameter_groups(
        model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None, 
        decoder_decay=None, decoder_list=(), no_lr_scale_list=[]):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        # Remove wrapped module to be compatible with FSDP
        name = name.replace("_fsdp_wrapped_module.", "")

        if not param.requires_grad:
            continue  # frozen weights

        # Assign weight decay values
        # Only norm and bias terms should have no decay
        # Previously, this checked if (param.shape) == 1 which is incompatible with FSDP which flattens all params
        if "norm." in name or ".norm" in name or name.endswith(".bias") or name.endswith(".lookup_table_weight") or name.endswith(".gamma") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        elif decoder_decay is not None and (name.startswith("decoder.") or name in decoder_list):
            group_name = "decoder_decay"
            this_weight_decay = decoder_decay
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        # Assign layer ID for LR scaling
        skip_scale = False
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
            if name in no_lr_scale_list:
                skip_scale = True
                group_name = f'{group_name}_no_lr_scale'
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None and not skip_scale:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    """
    Model can either be a single nn.Module, or a dictionary with {'model': model, 'balancer': balancer}.
    """
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    try:
        decoder_decay = args.decoder_decay
    except:
        decoder_decay = None
    try:
        no_lr_scale_list = args.no_lr_scale_list.split('-')
    except:
        no_lr_scale_list = []

    def get_parameters(m):
        if weight_decay and filter_bias_and_bn:
            skip = {}
            if skip_list is not None:
                skip = skip_list
            elif hasattr(m, 'no_weight_decay'):
                skip = m.no_weight_decay()
            decoder={}
            if hasattr(m, 'decoder_weight_decay'):
                decoder = m.decoder_weight_decay()
            parameters = get_parameter_groups(m, weight_decay, skip, get_num_layer, get_layer_scale, decoder_decay, decoder, no_lr_scale_list)
            wd = 0.
        else:
            parameters = m.parameters()
            wd = weight_decay
        return parameters, wd
    
    if isinstance(model, torch.nn.Module):
        parameters, weight_decay = get_parameters(model)
    elif isinstance(model, dict):
        print("WARNING: Weight decay assignment is skipped. All layers are assigned a weight decay value." )
        parameters = [
            {
                "params": [p for n, p in model['model'].named_parameters()
                        if p.requires_grad],
                "lr_scale": 1.,
            },
            {
                "params": [p for n, p in model['balancer'].named_parameters()
                        if p.requires_grad],
                "lr_scale": args.balancer_lr_scale,
            },
        ]

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    print("optimizer settings:", opt_args)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]

    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    return optimizer