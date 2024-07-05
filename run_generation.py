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
import os
import argparse
import time
import datetime
import copy
from pathlib import Path
import yaml
import json
import warnings

# PyTorch & friends
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torchvision.transforms.functional as TF 

# Metrics 
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal import CLIPScore

# Tokenizers (text & image modalities)
from tokenizers import Tokenizer
from fourm.vq.vqvae import VQVAE, DiVAE, VQControlNet

# 4M
from fourm.utils import load_safetensors
from fourm.models.fm import FM
from fourm.data.modality_info import MODALITY_INFO
from fourm.models.generate import GenerationSampler

# Local
import fourm.utils as utils
from fourm.data.modality_info import MODALITY_INFO, MODALITY_TRANSFORMS
from fourm.models.generate import build_chained_generation_schedules, init_empty_target_modality, init_full_input_modality
from fourm.data.masking import UnifiedMasking
from fourm.data.modality_transforms import UnifiedDataTransform, CropSettingsTransform
from fourm.data.multimodal_dataset_folder import MultiModalDatasetFolder
from fourm.data import PreTokenizedImageAugmenter, RandomCropImageAugmenter
from fourm.data.dataset_utils import SubsampleDatasetWrapper
from fourm.utils.generation_datasets import PartiPromptsDataset, EmptyDataset
from fourm.utils.generation import batch_to_device
from fourm.utils.plotting_utils import decode_dict, plot_conds_and_targets, save_conds_and_targets, denormalize

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

torch.set_grad_enabled(False)


def get_args(args=None):
    config_parser = parser = argparse.ArgumentParser(description='Generation Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                       help='YAML config file specifying default arguments')
    parser.add_argument('-dc', '--data_config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying validation data specific arguments')
    parser.add_argument('-gc', '--gen_config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying generation specific arguments')
    parser.add_argument('-src', '--sr_config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying super resolution specific arguments')

    parser = argparse.ArgumentParser('FourM generation script', add_help=False)    

    parser.add_argument('--run_name', type=str, default='auto')

    
    # Generation parameters
    parser.add_argument('--cond_domains', default='caption-det', type=str,
                        help='Conditioning domain names, separated by hyphen (default: %(default)s)')
    parser.add_argument('--target_domains', default='tok_clip-tok_normal-tok_rgb', type=str,
                        help='Target domain names, separated by hyphen. (default: %(default)s)')
    parser.add_argument('--tokens_per_target', default='196-196-196', type=str,
                        help='Number of tokens for each target modality. (default: %(default)s)')
    parser.add_argument('--autoregression_schemes', default='maskgit-maskgit-maskgit', type=str,
                        help='Scheme of autoregressive generation for each target modality. "maskgit", "roar" or "autoregressive" (default: %(default)s)')
    parser.add_argument('--decoding_steps', default='25-25-25', type=str,
                        help='Number of decoding steps for each target modality. (default: %(default)s)')
    parser.add_argument('--token_decoding_schedules', default='cosine-cosine-cosine', type=str,
                        help='Token decoding schedules for each target modality. (default: %(default)s)')
    parser.add_argument('--temps', default='5.0-1.0-1.0', type=str,
                        help='Starting temperature for each target modality. (default: %(default)s)')
    parser.add_argument('--temp_schedules', default='linear-linear-linear', type=str,
                        help='Temperature schedules for each target modality. (default: %(default)s)')
    parser.add_argument('--cfg_scales', default='4.0-4.0-4.0', type=str,
                        help='Classifier-free guidance scales for each target modality. (default: %(default)s)')
    parser.add_argument('--cfg_schedules', default='constant-constant-constant', type=str,
                        help='Classifier-free guidance schedules for each target modality. (default: %(default)s)')
    parser.add_argument('--cfg_grow_conditioning', action='store_true',
                        help='After every completed modality, add them to classifier-free guidance conditioning.')
    parser.add_argument('--no_cfg_grow_conditioning', action='store_false', dest='cfg_grow_conditioning',
                        help='Perform classifier-free guidance only on initial conditioning.')
    parser.set_defaults(cfg_grow_conditioning=True)
    parser.add_argument('--top_p', default=0.0, type=float,
                        help='top_p > 0.0: Keep the top tokens with cumulative probability >= top_p (a.k.a. nucleus filtering) (default: %(default)s)')
    parser.add_argument('--top_k', default=0.0, type=float,
                        help='top_k > 0: Keep only top k tokens with highest probability (a.k.a. top-k filtering) (default: %(default)s)')

    # Super resolution parameters
    parser.add_argument('--sr_cond_domains', default=None, type=str,
                        help='SuperRes: Conditioning domain names, separated by hyphen. If none, all base conditions and targets are used. (default: %(default)s)')
    parser.add_argument('--sr_target_domains', default='tok_clip@448-tok_rgb@448', type=str,
                        help='SuperRes: Target domain names, separated by hyphen. (default: %(default)s)')
    parser.add_argument('--sr_tokens_per_target', default='784', type=str,
                        help='SuperRes: Number of tokens for each target modality. (default: %(default)s)')
    parser.add_argument('--sr_autoregression_schemes', default='maskgit', type=str,
                        help='SuperRes: Scheme of autoregressive generation for each target modality. "maskgit", "roar" or "autoregressive" (default: %(default)s)')
    parser.add_argument('--sr_decoding_steps', default='8', type=str,
                        help='SuperRes: Number of decoding steps for each target modality. (default: %(default)s)')
    parser.add_argument('--sr_token_decoding_schedules', default='cosine', type=str,
                        help='SuperRes: Token decoding schedules for each target modality. (default: %(default)s)')
    parser.add_argument('--sr_temps', default='1.0', type=str,
                        help='SuperRes: Starting temperature for each target modality. (default: %(default)s)')
    parser.add_argument('--sr_temp_schedules', default='linear', type=str,
                        help='SuperRes: Temperature schedules for each target modality. (default: %(default)s)')
    parser.add_argument('--sr_cfg_scales', default='4.0', type=str,
                        help='SuperRes: Classifier-free guidance scales for each target modality. (default: %(default)s)')
    parser.add_argument('--sr_cfg_schedules', default='constant', type=str,
                        help='SuperRes: Classifier-free guidance schedules for each target modality. (default: %(default)s)')
    parser.add_argument('--sr_cfg_grow_conditioning', action='store_true',
                        help='SuperRes: After every completed modality, add them to classifier-free guidance conditioning.')
    parser.add_argument('--sr_no_cfg_grow_conditioning', action='store_false', dest='sr_cfg_grow_conditioning',
                        help='SuperRes: Perform classifier-free guidance only on initial conditioning.')
    parser.set_defaults(sr_cfg_grow_conditioning=True)
    parser.add_argument('--sr_top_p', default=0.0, type=float,
                        help='SuperRes: top_p > 0.0: Keep the top tokens with cumulative probability >= top_p (a.k.a. nucleus filtering) (default: %(default)s)')
    parser.add_argument('--sr_top_k', default=0.0, type=float,
                        help='SuperRes: top_k > 0: Keep only top k tokens with highest probability (a.k.a. top-k filtering) (default: %(default)s)')
    
    # Script parameters
    parser.add_argument('--num_samples', default=None,
                        help='Maximum number of samples to draw from the dataloader. (default: %(default)s)')
    parser.add_argument('--num_variations', default=1, type=int,
                        help='Number of variations to generate from each sample. (default: %(default)s)')
    parser.add_argument('--seed', default=0, type=int, help='Random seed ')
    
    # Tokenizer settings
    parser.add_argument('--detokenizer_steps', default=25, type=int,
                        help='Number of DDPM/DDIM steps for decoding with diffusion-based tokenizers. (default: %(default)s)')
    parser.add_argument('--rgb_tok_id', default=None, type=str,
                        help='RGB tokenizer ID (default: %(default)s)')
    parser.add_argument('--depth_tok_id', default=None, type=str,
                        help='Depth tokenizer ID (default: %(default)s)')
    parser.add_argument('--normal_tok_id', default=None, type=str,
                        help='Normal tokenizer ID (default: %(default)s)')
    parser.add_argument('--edges_tok_id', default=None, type=str,
                        help='Edges tokenizer ID (default: %(default)s)')
    parser.add_argument('--semseg_tok_id', default=None, type=str,
                        help='Semseg tokenizer ID (default: %(default)s)')
    parser.add_argument('--clip_tok_id', default=None, type=str,
                        help='CLIP tokenizer ID (default: %(default)s)')
    parser.add_argument('--dinov2_tok_id', default=None, type=str,
                        help='DINOv2 tokenizer ID (default: %(default)s)')
    parser.add_argument('--imagebind_tok_id', default=None, type=str,
                        help='ImageBind tokenizer ID (default: %(default)s)')
    parser.add_argument('--dinov2_glob_tok_id', default=None, type=str,
                        help='DINOv2 global tokenizer ID (default: %(default)s)')
    parser.add_argument('--imagebind_glob_tok_id', default=None, type=str,
                        help='ImageBind global tokenizer ID (default: %(default)s)')
    parser.add_argument('--sam_instance_tok_id', default=None, type=str,
                        help='SAM instance tokenizer ID (default: %(default)s)')
    parser.add_argument('--human_poses_tok_id', default=None, type=str,
                        help='Human poses tokenizer ID (default: %(default)s)')
    parser.add_argument('--text_tok_path', default='fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json', type=str,
                        help='Text tokenizer path (default: %(default)s)')
    
    # ControlNet parameters
    parser.add_argument('--activate_controlnet', action='store_true',
                        help='When enabled, RGB detokenizer will be replaced by RGB ControlNet.')
    parser.add_argument('--no_activate_controlnet', action='store_false', dest='activate_controlnet')
    parser.set_defaults(activate_controlnet=False)
    parser.add_argument('--controlnet_id', default=None, type=str,
                        help='RGB ControlNet ID (default: %(default)s)')
    parser.add_argument('--controlnet_guidance_scale', default=2.5, type=float,
                        help='RGB ControlNet guidance scale (default: %(default)s)')
    parser.add_argument('--controlnet_cond_scale', default=0.8, type=float,
                        help='RGB ControlNet conditioning scale (default: %(default)s)')
    
    # Model parameters
    parser.add_argument('--model', default=None, type=str, metavar='MODEL',
                        help='4M model: Hugging Face Hub ID, or path to local safetensors checkpoint (default: %(default)s)')
    parser.add_argument('--sr_model', default=None, type=str, metavar='MODEL',
                        help='Superres model: Hugging Face Hub ID, or path to local safetensors checkpoint (default: %(default)s)')
    parser.add_argument('--image_size', default=224, type=int,
                        help='Image size. (default: %(default)s)')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Base patch size for image-like modalities (default: %(default)s)')
    
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float16', 'bfloat16', 'float32', 'bf16', 'fp16', 'fp32'],
                        help='Data type (default: %(default)s')
    # Data
    parser.add_argument('--data_path', default='/mnt/datasets/cc12_multitask_224/val', 
                        help='Path to dataset (default: %(default)s)')
    parser.add_argument('--data_name', default='', type=str,
                        help='Name of dataset, used for wandb and output folder. (default: %(default)s)')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--parti_prompts_t5_embs', default=None, type=str,
                        help="(Optional) path to pre-computed T5 embeddings for PartiPrompts (in .npz format)")
    
    # Misc.
    parser.add_argument('--s3_endpoint', default='', type=str, help='S3 endpoint URL')
    parser.add_argument('--s3_path', default='', type=str, help='S3 path to model')
    parser.add_argument('--image_size_metrics', default=256, type=int,
                        help='Image size for computing FID, Inception, and CLIP metrics. (default: %(default)s)')
    parser.add_argument('--name', default='', type=str,
                        help='wandb and folder name (default: %(default)s)')
    parser.add_argument('--sr_name', default='', type=str,
                        help='SR wandb and folder name (default: %(default)s)')
    parser.add_argument('--output_dir', default='',
                        help='Path where to save, empty for no saving')
    parser.add_argument('--num_log_images', default=100,
                        help='Number of images to log (default: %(default)s)')
    parser.add_argument('--save_all_outputs', action='store_true',
                        help='Save all conditioning and target modalities for all drawn samples as individual files.')
    parser.add_argument('--no_save_all_outputs', action='store_false', dest='save_all_outputs',
                        help='Do not save any outputs.')
    parser.set_defaults(save_all_outputs=False)
    
    # Wandb logging
    parser.add_argument('--log_wandb', default=False, action='store_true',
                        help='Log training and validation metrics to wandb')
    parser.add_argument('--no_log_wandb', action='store_false', dest='log_wandb')
    parser.set_defaults(log_wandb=False)
    parser.add_argument('--wandb_project', default=None, type=str,
                        help='Project name on wandb')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='User or team name on wandb')
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='Run name on wandb')
    parser.add_argument('--wandb_mode', default='online', type=str,
                        help='Wandb mode')
    parser.add_argument('--show_user_warnings', default=False, action='store_true')

    # GPU / Distributed parameters
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training / testing')
    parser.add_argument('--dist_gen', action='store_true', default=False,
                        help='Enabling distributed generation')
    parser.add_argument('--no_dist_gen', action='store_false', dest='dist_gen',
                        help='Disabling distributed generation')
    parser.set_defaults(dist_gen=True)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Parse config file if there is one
    args_config, remaining = config_parser.parse_known_args(args)

    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    if args_config.data_config:
        with open(args_config.data_config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    if args_config.gen_config:
        with open(args_config.gen_config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    if args_config.sr_config:
        with open(args_config.sr_config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    #The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Add the config paths if given
    args.config_path = args_config.config
    args.data_config_path = args_config.data_config
    args.gen_config_path = args_config.gen_config
    args.sr_config_path = args_config.sr_config
    
    return args


def truncate_caption_for_clip(caption, clip_tokenizer, max_tokens=60):
    seq_trunc = clip_tokenizer.encode(caption)
    seq_trunc = seq_trunc[:max_tokens-1] + [seq_trunc[-1]]
    cap_trunc = clip_tokenizer.decode(seq_trunc)
    return caption[:len(cap_trunc)]

def string_to_list(input_string, dtype=float, delim='-'):
    """
    Convert a string separated by hyphens into a list of a given data type, 
    replacing invalid values with None.
    
    Args:
        input_string (str): The input string to convert.
        dtype (type): The target data type for conversion. Default is float.
        delim (str): The delimiter used to separate values in the string. Default is '-'.

    Returns:
        list: A list of values in the specified data type or None for invalid values.
    """
    if input_string is None:
        return [None]
    
    if isinstance(input_string, float) or isinstance(input_string, int):
        return [input_string]
    
    def try_cast(item, dtype):
        try:
            return dtype(item)
        except ValueError:
            return None

    return [try_cast(item, dtype) for item in input_string.split(delim)]

def repeat_if_necessary(lst, n):
        return lst * n if len(lst) == 1 else lst        

def load_model(model_id, model_class, device):
    if model_id is None:
        model = None
    elif model_id.endswith('.safetensors'):
        ckpt, config = load_safetensors(model_id)
        model = model_class(config=config)
        model.load_state_dict(ckpt)
    else:
        model = model_class.from_pretrained(model_id)
    return model.eval().to(device)

def load_tokenizers(args, device):
    toks = {}

    # RGB tokenizer
    if args.rgb_tok_id:
        toks['tok_rgb'] = load_model(args.rgb_tok_id, DiVAE, device)

    # Optional RGB ControlNet
    if args.controlnet_id:
        toks['controlnet'] = load_model(args.controlnet_id, VQControlNet, device)

    # Depth tokenizer
    if args.depth_tok_id:
        toks['tok_depth'] = load_model(args.depth_tok_id, DiVAE, device)

    # Normal tokenizer
    if args.normal_tok_id:
        toks['tok_normal'] = load_model(args.normal_tok_id, DiVAE, device)

    # Edges tokenizer
    if args.edges_tok_id:
        toks['tok_canny_edge'] = load_model(args.edges_tok_id, DiVAE, device)
        toks['tok_sam_edge'] = toks['tok_canny_edge']

    # Semseg tokenizer
    if args.semseg_tok_id:
        toks['tok_semseg'] = load_model(args.semseg_tok_id, VQVAE, device)

    # CLIP tokenizer
    if args.clip_tok_id:
        toks['tok_clip'] = load_model(args.clip_tok_id, VQVAE, device)

    # DINOv2 tokenizer
    if args.dinov2_tok_id:
        toks['tok_dinov2'] = load_model(args.dinov2_tok_id, VQVAE, device)

    # ImageBind tokenizer
    if args.imagebind_tok_id:
        toks['tok_imagebind'] = load_model(args.imagebind_tok_id, VQVAE, device)

    # DINOv2 global tokenizer
    if args.dinov2_glob_tok_id:
        toks['tok_dinov2_global'] = load_model(args.dinov2_glob_tok_id, VQVAE, device)

    # ImageBind global tokenizer
    if args.imagebind_glob_tok_id:
        toks['tok_imagebind_global'] = load_model(args.imagebind_glob_tok_id, VQVAE, device)

    # SAM instances
    if args.sam_instance_tok_id:
        toks['sam_instance'] = load_model(args.sam_instance_tok_id, VQVAE, device)

    # Human poses
    if args.human_poses_tok_id:
        toks['tok_pose'] = load_model(args.human_poses_tok_id, VQVAE, device)

    return toks

def get_dataset(args, text_tokenizer):
    
    # For unconditional generation
    if len(args.cond_domains) == 0:
        args.loaded_domains = args.cond_domains
        dataset = EmptyDataset(dataset_size=args.num_samples)
    
    # For caption->X generation using Parti Prompts
    elif args.data_path == 'parti_prompts':
        llm_embedder = None
        args.loaded_domains = args.cond_domains
        args.parti_prompts_t5_embs = None

        dataset = PartiPromptsDataset(text_tokenizer, max_length=128, parti_prompts_t5_embs=args.parti_prompts_t5_embs, llm_embedder=llm_embedder)
    
     # Otherwise, construct CC12M/IN1K-like pre-tokenized dataset
    else:
        # Also load RGB (for det augmentation and FID calculation)
        args.loaded_domains = sorted(list(set(args.cond_domains) | set(['rgb'])))

        modality_transforms = MODALITY_TRANSFORMS

        modality_info = {mod: MODALITY_INFO[mod] for mod in args.loaded_domains}
        # Max tokens
        for k in modality_info:
            num_patches = (args.image_size // args.patch_size) ** 2
            if modality_info[k]['type'] == 'img':
                modality_info[k]['max_tokens'] = num_patches
        # Dirichlet concentration parameter (Alpha)
        for k in modality_info:
            modality_info[k]["input_alphas"] = [0.]
            modality_info[k]["target_alphas"] = [0.]
            modality_info[k]["keep"] = ['all']

        if 'tok' not in '-'.join(args.loaded_domains):
            image_augmenter = RandomCropImageAugmenter(
                target_size=args.image_size, hflip=False, 
                crop_scale=(1.0,1.0), crop_ratio=(1.0,1.0)
            )
        else:
            image_augmenter = PreTokenizedImageAugmenter(target_size=args.image_size, no_aug=True)
            modality_transforms["crop_settings"] = CropSettingsTransform()
            args.loaded_domains.append("crop_settings")

        transform = transforms.Compose([
            UnifiedDataTransform(transforms_dict=modality_transforms, image_augmenter=image_augmenter),
            UnifiedMasking(
                modality_info=modality_info, text_tokenizer=text_tokenizer,
                input_tokens_range=512, target_tokens_range=512
            ),
        ])

        modality_paths = {mod: modality_info[mod]['path'] for mod in modality_info if modality_info[mod].get('path', None) is not None}
        
        dataset = MultiModalDatasetFolder(
            args.data_path, args.loaded_domains, modality_paths=modality_paths,  
            modality_transforms=modality_transforms, transform=transform
        )
    
    # Subsample dataset if needed
    dataset = SubsampleDatasetWrapper(dataset, dataset_size=args.num_samples, seed=0, return_orig_idx=True)

    return dataset


def create_superres_input(out_dict, sr_cond_domains, sr_target_domains, sr_tokens_per_target, text_tokenizer, device):
    superres_sample = {}

    # Low-res condition and generated targets become condition for super resolution
    for domain in sr_cond_domains:
        superres_sample[domain] = out_dict[domain]

    # Initialize input modalities
    for cond_mod in sr_cond_domains:
        superres_sample = init_full_input_modality(superres_sample, MODALITY_INFO, cond_mod, device, eos_id=text_tokenizer.token_to_id("[EOS]"))
        
    # Initialize target modalities
    for target_mod, ntoks in zip(sr_target_domains, sr_tokens_per_target):
        superres_sample = init_empty_target_modality(superres_sample, MODALITY_INFO, target_mod, 1, ntoks, device)
        
    return superres_sample


def main(args):
    args = copy.deepcopy(args)
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    args.seed = args.seed + utils.get_rank()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # random.seed(args.seed)

    cudnn.benchmark = True

    if not args.show_user_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)

    if args.dtype in ['float16', 'fp16']:
        dtype = torch.float16
    elif args.dtype in ['bfloat16', 'bf16']:
        dtype = torch.bfloat16
    elif args.dtype in ['float32', 'fp32']:
        dtype = torch.float32
    else:
        raise ValueError(f"Invalid dtype: {args.dtype}")
    
    if args.data_name == 'auto':
        args.data_name = Path(args.data_config_path).stem
    if args.name == 'auto':
        args.name = Path(args.gen_config_path).stem
    if args.sr_name == 'auto':
        args.sr_name = Path(args.sr_config_path).stem

    # Output directory
    args.output_dir = os.path.join(args.output_dir, args.data_name, f'{args.name}--{args.sr_name}' if args.sr_name else args.name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare args
    delim = '-'

    # Generation parameters
    args.cond_domains = sorted(list(string_to_list(args.cond_domains, dtype=str, delim=delim)))
    args.target_domains = string_to_list(args.target_domains, dtype=str, delim=delim)
    args.all_domains = sorted(list(set(args.cond_domains) | set(args.target_domains)))
    args.loaded_domains = sorted(list(set(args.cond_domains) | set(['rgb'])))
    n_targets = len(args.target_domains)
    args.tokens_per_target = repeat_if_necessary(string_to_list(args.tokens_per_target, dtype=int, delim=delim), n_targets)
    args.autoregression_schemes = repeat_if_necessary(string_to_list(args.autoregression_schemes, dtype=str, delim=delim), n_targets)
    args.decoding_steps = repeat_if_necessary(string_to_list(args.decoding_steps, dtype=int, delim=delim), n_targets)
    args.token_decoding_schedules = repeat_if_necessary(string_to_list(args.token_decoding_schedules, dtype=str, delim=delim), n_targets)
    args.temps = repeat_if_necessary(string_to_list(args.temps, dtype=float, delim=delim), n_targets)
    args.temp_schedules = repeat_if_necessary(string_to_list(args.temp_schedules, dtype=str, delim=delim), n_targets)
    args.cfg_scales = repeat_if_necessary(string_to_list(args.cfg_scales, dtype=float, delim=delim), n_targets)
    args.cfg_schedules = repeat_if_necessary(string_to_list(args.cfg_schedules, dtype=str, delim=delim), n_targets)

    # Super-resolution parameters
    if args.sr_cond_domains is None:
        args.sr_cond_domains = args.cond_domains + args.target_domains
    else:
        args.sr_cond_domains = sorted(list(string_to_list(args.sr_cond_domains, dtype=str, delim=delim)))
    args.sr_target_domains = string_to_list(args.sr_target_domains, dtype=str, delim=delim)
    args.sr_all_domains = sorted(list(set(args.sr_cond_domains) | set(args.sr_target_domains)))
    sr_n_targets = len(args.sr_target_domains)
    args.sr_tokens_per_target = repeat_if_necessary(string_to_list(args.sr_tokens_per_target, dtype=int, delim=delim), sr_n_targets)
    args.sr_autoregression_schemes = repeat_if_necessary(string_to_list(args.sr_autoregression_schemes, dtype=str, delim=delim), sr_n_targets)
    args.sr_decoding_steps = repeat_if_necessary(string_to_list(args.sr_decoding_steps, dtype=int, delim=delim), sr_n_targets)
    args.sr_token_decoding_schedules = repeat_if_necessary(string_to_list(args.sr_token_decoding_schedules, dtype=str, delim=delim), sr_n_targets)
    args.sr_temps = repeat_if_necessary(string_to_list(args.sr_temps, dtype=float, delim=delim), sr_n_targets)
    args.sr_temp_schedules = repeat_if_necessary(string_to_list(args.sr_temp_schedules, dtype=str, delim=delim), sr_n_targets)
    args.sr_cfg_scales = repeat_if_necessary(string_to_list(args.sr_cfg_scales, dtype=float, delim=delim), sr_n_targets)
    args.sr_cfg_schedules = repeat_if_necessary(string_to_list(args.sr_cfg_schedules, dtype=str, delim=delim), sr_n_targets)

    # Load text tokenizer
    text_tokenizer = Tokenizer.from_file(args.text_tok_path)

    # Load image tokenizers
    tokenizers = load_tokenizers(args, device)

    # Load model & define sampler
    model = load_model(args.model, FM, device)
    gen_sampler= GenerationSampler(model)

    # Load super-resolution model if so specified
    model_sr = load_model(args.sr_model, FM, device)
    gen_sampler_sr = GenerationSampler(model_sr) if model_sr is not None else None    

    # Get dataset
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    dataset = get_dataset(args, text_tokenizer)
    if args.dist_gen:
        if len(dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                'This will slightly alter validation results as extra duplicate entries are added to achieve '
                'equal num of samples per-process.')
        data_sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        data_sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=data_sampler,
        batch_size=1, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
    )

    # Logging
    if global_rank == 0 and args.log_wandb:
        # Edit run name and add tags
        args.wandb_tags = [args.data_name, args.name, args.wandb_run_name]
        if args.sr_name:
            args.wandb_tags.append(args.sr_name)
        args.wandb_run_name = f"{args.name}--{args.sr_name}--{args.data_name}--{args.wandb_run_name}"
        log_writer = utils.WandbLogger(args)
        log_writer.set_step(0)
    else:
        log_writer = None

    print('\nArguments:')
    print(args)
    print('')

    print('Starting generation...')
    start_time = time.time()

    # Measure generation statistics & save samples
    gen_stats = generate(gen_sampler, gen_sampler_sr, tokenizers, text_tokenizer, data_loader, device, dtype, args)

    if log_writer is not None:
        log_writer.update(gen_stats)

    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, "log_eval.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(gen_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Done! Total generation time {} on device {}'.format(total_time_str, device))
    torch.cuda.empty_cache()


@torch.no_grad()
def generate(gen_sampler, gen_sampler_sr, tokenizers, text_tokenizer, data_loader, device, dtype, args):

    # Set up generation schedule
    schedule = build_chained_generation_schedules(
        cond_domains=args.cond_domains, 
        target_domains=args.target_domains,
        tokens_per_target=args.tokens_per_target,
        autoregression_schemes=args.autoregression_schemes, 
        decoding_steps=args.decoding_steps, 
        token_decoding_schedules=args.token_decoding_schedules,
        temps =args.temps,
        temp_schedules=args.temp_schedules,
        cfg_scales=args.cfg_scales, 
        cfg_schedules=args.cfg_schedules,
        cfg_grow_conditioning=args.cfg_grow_conditioning, 
    )

    # Set up super resolution schedule
    sr_schedule = build_chained_generation_schedules(
        cond_domains=args.sr_cond_domains, 
        target_domains=args.sr_target_domains,
        tokens_per_target=args.sr_tokens_per_target,
        autoregression_schemes=args.sr_autoregression_schemes, 
        decoding_steps=args.sr_decoding_steps, 
        token_decoding_schedules=args.sr_token_decoding_schedules,
        temps =args.sr_temps,
        temp_schedules=args.sr_temp_schedules,
        cfg_scales=args.sr_cfg_scales, 
        cfg_schedules=args.sr_cfg_schedules,
        cfg_grow_conditioning=args.sr_cfg_grow_conditioning, 
    ) if gen_sampler_sr is not None else None

    # Set up metric loggers
    fid_metric, inception_metric, clip_metric = None, None, None
    if 'tok_rgb' in args.target_domains:
        inception_metric = InceptionScore(
            feature='logits_unbiased', splits=10, normalize=False,
            sync_on_compute=True
        ).to(device)
        if 'rgb' in args.loaded_domains:
            fid_metric = FrechetInceptionDistance(
                feature=2048, reset_real_features=True, 
                normalize=False, sync_on_compute=True
            ).to(device)
        if 'caption' in args.cond_domains:
            clip_metric = CLIPScore(
                model_name_or_path="openai/clip-vit-large-patch14", 
                sync_on_compute=True
            ).to(device)

    # For super resolution as well (if it is performed)
    fid_metric_sr, inception_metric_sr, clip_metric_sr = None, None, None
    if gen_sampler_sr is not None and 'tok_rgb@448' in args.sr_target_domains:
        inception_metric_sr = InceptionScore(
            feature='logits_unbiased', splits=10, normalize=False,
            sync_on_compute=True
        ).to(device)
        if 'rgb' in args.loaded_domains:
            fid_metric_sr = FrechetInceptionDistance(
                feature=2048, reset_real_features=True, 
                normalize=False, sync_on_compute=True
            ).to(device)
        if 'caption' in args.cond_domains:
            clip_metric_sr = CLIPScore(
                model_name_or_path="openai/clip-vit-large-patch14", 
                sync_on_compute=True
            ).to(device)


    metric_logger = utils.MetricLogger(delimiter="  ")

    logged_images_count = 0

    for sample, sample_idx in metric_logger.log_every(data_loader, print_freq=1, header='Generation:'):
        sample_idx = sample_idx[0].item()

        # Sample to device
        sample = batch_to_device(sample, device, domains=args.loaded_domains)

        # Update FID metric with a sample from the real distribution
        if fid_metric is not None or fid_metric_sr is not None:
            rgb_real = (255 * denormalize(sample['rgb']['tensor'])).to(torch.uint8)
            rgb_real = TF.resize(rgb_real, size=args.image_size_metrics)
            if fid_metric is not None:
                fid_metric.update(rgb_real, real=True)
            if fid_metric_sr is not None:
                fid_metric_sr.update(rgb_real, real=True)

        # Remove RGB if it is not used as an input (just loaded to make det dataloading happy and for metrics)
        for domain in args.loaded_domains:
            if domain not in args.cond_domains and domain in sample:
                del sample[domain]

        # Initialize input modalities
        for cond_mod in args.cond_domains:
            sample = init_full_input_modality(sample, MODALITY_INFO, cond_mod, device, eos_id=text_tokenizer.token_to_id("[EOS]"))

        # Initialize target modalities
        for target_mod, ntoks in zip(args.target_domains, args.tokens_per_target):
            sample = init_empty_target_modality(sample, MODALITY_INFO, target_mod, 1, ntoks, device)
            
        
        dec_dicts = []
        dec_dicts_sr = []

        # Draw several samples using the same conditioning
        for i in range(args.num_variations):
            with torch.cuda.amp.autocast(dtype=dtype, enabled=dtype != torch.float32):
                out_dict = gen_sampler.generate(
                    sample, schedule, text_tokenizer=text_tokenizer, verbose=False, 
                    seed=utils.generate_seed(args.seed, sample_idx, i),
                    top_p=args.top_p, top_k=args.top_k
                )

            # Decode tokens into images/text
            dec_dict = decode_dict(
                out_dict, tokenizers, text_tokenizer, 
                image_size=args.image_size, patch_size=args.patch_size, 
                decoding_steps=args.detokenizer_steps, 
                activate_controlnet=args.activate_controlnet,
                controlnet_guidance_scale=args.controlnet_guidance_scale,
                controlnet_cond_scale=args.controlnet_cond_scale,
            )
            dec_dicts.append(dec_dict)

            # Update metrics
            if inception_metric is not None:
                rgb_pred = TF.to_tensor(255 * dec_dict['tok_rgb']).to(dtype=torch.uint8, device=device).unsqueeze(0)
                rgb_pred = TF.resize(rgb_pred, size=args.image_size_metrics)
                inception_metric.update(rgb_pred)
                if fid_metric is not None:
                    fid_metric.update(rgb_pred, real=False)
                if clip_metric is not None:
                    caption_trunc = truncate_caption_for_clip(dec_dict['caption'][0], clip_metric.processor.tokenizer)
                    clip_metric.update(rgb_pred, caption_trunc)

            # Super-resolution
            if gen_sampler_sr is not None:
                with torch.cuda.amp.autocast(dtype=dtype, enabled=dtype != torch.float32):
                    sample_sr = create_superres_input(
                        out_dict, args.sr_cond_domains, args.sr_target_domains, 
                        args.sr_tokens_per_target, text_tokenizer, device
                    )
                    out_dict_sr = gen_sampler_sr.generate(
                        sample_sr, sr_schedule, text_tokenizer=text_tokenizer, verbose=False, 
                        seed=utils.generate_seed(args.seed, sample_idx, i),
                        top_p=args.sr_top_p, top_k=args.sr_top_k,
                    )

                # Decode tokens into images/text
                dec_dict_sr = decode_dict(
                    out_dict_sr, tokenizers, text_tokenizer, 
                    image_size=448, patch_size=args.patch_size, 
                    decoding_steps=args.detokenizer_steps,
                    activate_controlnet=args.activate_controlnet,
                    controlnet_guidance_scale=args.controlnet_guidance_scale,
                    controlnet_cond_scale=args.controlnet_cond_scale,
                )
                dec_dicts_sr.append(dec_dict_sr)

                # Update superres metrics
                if inception_metric_sr is not None:
                    rgb_pred = TF.to_tensor(255 * dec_dict_sr['tok_rgb@448']).to(dtype=torch.uint8, device=device).unsqueeze(0)
                    rgb_pred = TF.resize(rgb_pred, size=args.image_size_metrics)
                    inception_metric_sr.update(rgb_pred)
                    if fid_metric_sr is not None:
                        fid_metric_sr.update(rgb_pred, real=False)
                    if clip_metric_sr is not None:
                        caption_trunc = truncate_caption_for_clip(dec_dict['caption'][0], clip_metric_sr.processor.tokenizer)
                        clip_metric_sr.update(rgb_pred, caption_trunc)
            

        # Save all-in-one plot
        if args.num_log_images == 'all' or (utils.is_main_process() and logged_images_count < int(args.num_log_images)):
            plot_conds_and_targets(
                args.cond_domains, args.target_domains, dec_dicts, 
                save_path=os.path.join(args.output_dir, 'plots', f'{sample_idx:06d}.jpg')
            )
            for sr_idx, sr_dec_dict in enumerate(dec_dicts_sr):
                plot_conds_and_targets(
                    args.sr_cond_domains, args.sr_target_domains, [sr_dec_dict], 
                    save_path=os.path.join(args.output_dir, 'plots', f'{sample_idx:06d}_sr{sr_idx}.jpg')
                )
            logged_images_count += 1

        # Save each modality separately
        if args.save_all_outputs:
            save_conds_and_targets(
                args.cond_domains, args.target_domains, dec_dicts, 
                save_dir=args.output_dir, sample_idx=sample_idx
            )

    # Compute and log metrics
    results = {}

    if inception_metric is not None:
        inception_mean, inception_std = inception_metric.compute()
        results['inception_mean'] = inception_mean.item()
        results['inception_std'] = inception_std.item()
    if fid_metric is not None:
        fid = fid_metric.compute().item()
        results['fid'] = fid
    if clip_metric is not None:
        clip_score = clip_metric.compute().item()
        results['clip_score'] = clip_score

    if inception_metric_sr is not None:
        inception_mean_sr, inception_std_sr = inception_metric_sr.compute()
        results['inception_mean_sr'] = inception_mean_sr.item()
        results['inception_std_sr'] = inception_std_sr.item()
    if fid_metric_sr is not None:
        fid_sr = fid_metric_sr.compute().item()
        results['fid_sr'] = fid_sr
    if clip_metric_sr is not None:
        clip_score_sr = clip_metric_sr.compute().item()
        results['clip_score_sr'] = clip_score_sr

    metric_logger.update(**results)
    # Gather the stats from all processes (they should already be the same since we sync the torcheval metrics after every step)
    metric_logger.synchronize_between_processes()
    print("Generation results:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    args = get_args()

    utils.setup_run_name(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
