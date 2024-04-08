import sys, os
import torch

from .vqvae import VQ, VQVAE, DiVAE, VQControlNet
from .scheduling import *


def get_image_tokenizer(tokenizer_id: str, 
                        tokenizers_root: str = './tokenizer_ckpts', 
                        encoder_only: bool = False, 
                        device: str = 'cuda', 
                        verbose: bool = True,
                        return_None_on_fail: bool = False,):
    """
    Load a pretrained image tokenizer from a checkpoint.

    Args:
        tokenizer_id (str): ID of the tokenizer to load (name of the checkpoint file without ".pth").
        tokenizers_root (str): Path to the directory containing the tokenizer checkpoints.
        encoder_only (bool): Set to True to load only the encoder part of the tokenizer.
        device (str): Device to load the tokenizer on.
        verbose (bool): Set to True to print load_state_dict warning/success messages
        return_None_on_fail (bool): Set to True to return None if the tokenizer fails to load (e.g. doesn't exist)

    Returns:
        model (nn.Module): The loaded tokenizer.
    """
    if return_None_on_fail and not os.path.exists(os.path.join(tokenizers_root, f'{tokenizer_id}.pth')):
        return None
    
    if verbose:
        print(f'Loading tokenizer {tokenizer_id} ... ', end='')
    
    ckpt = torch.load(os.path.join(tokenizers_root, f'{tokenizer_id}.pth'), map_location='cpu')

    # Handle renamed arguments
    if 'CLIP' in ckpt['args'].domain or 'DINO' in ckpt['args'].domain or 'ImageBind' in ckpt['args'].domain:
        ckpt['args'].patch_proj = False
    elif 'sam' in ckpt['args'].domain:
        ckpt['args'].input_size_min = ckpt['args'].mask_size
        ckpt['args'].input_size_max = ckpt['args'].mask_size
        ckpt['args'].input_size = ckpt['args'].mask_size
        
    ckpt['args'].quant_type = getattr(ckpt['args'], 'quantizer_type', None)
    ckpt['args'].enc_type = getattr(ckpt['args'], 'encoder_type', None)
    ckpt['args'].dec_type = getattr(ckpt['args'], 'decoder_type', None)
    ckpt['args'].image_size = getattr(ckpt['args'], 'input_size', None) or getattr(ckpt['args'], 'input_size_max', None)
    ckpt['args'].image_size_enc = getattr(ckpt['args'], 'input_size_enc', None)
    ckpt['args'].image_size_dec = getattr(ckpt['args'], 'input_size_dec', None)
    ckpt['args'].image_size_sd = getattr(ckpt['args'], 'input_size_sd', None)
    ckpt['args'].ema_decay = getattr(ckpt['args'], 'quantizer_ema_decay', None)
    ckpt['args'].enable_xformer = getattr(ckpt['args'], 'use_xformer', None)
    if 'cls_emb.weight' in ckpt['model']:
        ckpt['args'].n_labels, ckpt['args'].n_channels = n_labels, n_channels = ckpt['model']['cls_emb.weight'].shape
    elif 'encoder.linear_in.weight' in ckpt['model']:
        ckpt['args'].n_channels = ckpt['model']['encoder.linear_in.weight'].shape[1]
    else:
        ckpt['args'].n_channels = ckpt['model']['encoder.proj.weight'].shape[1]
    ckpt['args'].sync_codebook = False
    
    if encoder_only:
        model_type = VQ
        ckpt['model'] = {k: v for k, v in ckpt['model'].items() if 'decoder' not in k and 'post_quant_proj' not in k}
    else:
        # TODO: Add the model type to the checkpoint when training so we can avoid this hackery
        if any(['controlnet' in k for k in ckpt['model'].keys()]):
            ckpt['args'].model_type = 'VQControlNet'
        elif hasattr(ckpt['args'], 'beta_schedule'):
            ckpt['args'].model_type = 'DiVAE'
        else:
            ckpt['args'].model_type = 'VQVAE'
        model_type = getattr(sys.modules[__name__], ckpt['args'].model_type)
    model = model_type(**vars(ckpt['args']))
    
    msg = model.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(msg)

    return model.to(device).eval(), ckpt['args']
