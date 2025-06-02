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
# Based on the timm code base
# https://github.com/huggingface/pytorch-image-models
# --------------------------------------------------------
import io
import os
import json
from yaml import safe_load, YAMLError
from pathlib import Path
from safetensors.torch import load as load_st

import torch

from .dist import save_on_main, is_main_process
from .timm.model import get_state_dict
from .s3_utils import save_on_s3


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def load_state_dict(model, state_dict, prefix='', ignore_missing=''):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, loss_balancer=None, model_ema=None, ckpt_name=None, use_s3=False, all_nodes=False):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    ckpt_name = ckpt_name or epoch_name

    # Only create the save_dict on the main process, unless all_nodes is set to True
    if is_main_process() or (all_nodes and args.gpu == 0): 
        checkpoint_path = os.path.join(output_dir, f'checkpoint-{ckpt_name}.pth')

        to_save = {
            'model': model_without_ddp.state_dict(),
            'epoch': epoch,
            'args': args,
            'scaler': loss_scaler.state_dict(),
        }

        if optimizer is not None:
            to_save['optimizer'] = optimizer.state_dict()

        if loss_balancer is not None:
            to_save['loss_balancer'] = loss_balancer.state_dict()

        if model_ema is not None:
            to_save['model_ema'] = get_state_dict(model_ema)

        save_on_main(to_save, checkpoint_path)
        
        if use_s3: 
            s3_path = os.path.join(args.s3_save_dir, f'checkpoint-{ckpt_name}.pth')
            save_on_s3(checkpoint_path, s3_path, args.s3_endpoint)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    # torch.amp
    if args.auto_resume and len(args.resume) == 0:
        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu')
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1

            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")

        if hasattr(args, 'model_ema') and args.model_ema:
            _load_checkpoint_for_ema(model_ema, {'state_dict_ema': checkpoint['model_ema']})
            print("With EMA!")


MAX_LEN_YAML_PARSE = 10_000

def safe_parse_metadata(metadata_str):
    metadata = {}
    for k, v in metadata_str.items():
        if not isinstance(v, str) or len(v) > MAX_LEN_YAML_PARSE:
            metadata[k] = v
            continue
        try:
            parsed = safe_load(v.replace('None', 'null'))
            metadata[k] = parsed
        except YAMLError:
            metadata[k] = v
    return metadata


def load_safetensors(safetensors_path, return_metadata=True):
    with open(safetensors_path, 'rb') as f:
        data = f.read()

    tensors = load_st(data)

    if not return_metadata:
        return tensors
    
    n_header = data[:8]
    n = int.from_bytes(n_header, "little")
    metadata_bytes = data[8 : 8 + n]
    header = json.loads(metadata_bytes)
    metadata = header.get("__metadata__", {})
    metadata = safe_parse_metadata(metadata)

    return tensors, metadata