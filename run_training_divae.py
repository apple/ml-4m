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
import sys
sys.path.insert(0,'..')
import argparse
import datetime
import wandb
from wandb import AlertLevel
import json
import math
import os
import io
import re
import time
import warnings
from pathlib import Path
from typing import Iterable, List, Set, Dict, Optional, Union, Callable
import yaml
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from einops import rearrange, repeat

import webdataset as wds
from webdataset.handlers import reraise_exception
import boto3
from boto3.s3.transfer import TransferConfig

# Metrics
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.inception import InceptionScore

from diffusers.schedulers.scheduling_utils import SchedulerMixin
import diffusers.schedulers as diffusers_schedulers
from fourm.vq.scheduling import DDPMScheduler, DDIMScheduler

import fourm.utils as utils
from fourm.data import build_wds_divae_dataloader
from fourm.data import RandomCropImageAugmenter, CenterCropImageAugmenter
import fourm.utils.data_constants as data_constants
from fourm.utils import denormalize
from fourm.utils.optim_factory import create_optimizer
from fourm.utils import to_2tuple
from fourm.utils import NativeScalerWithGradNormCount as NativeScaler
from fourm.utils import ModelEmaV2 as ModelEma
from fourm.vq.vqvae import DiVAE
from fourm.vq.vq_utils import compute_codebook_usage

from fourm.data.modality_info import MODALITY_INFO, MODALITY_TRANSFORMS_DIVAE
from fourm.data.modality_transforms import UnifiedDataTransform, RGBTransform, NormalTransform
from fourm.data.multimodal_dataset_folder import MultiModalDatasetFolder


def unwrap_model(model: Union[nn.Module, DDP]) -> nn.Module:
    """Retrieves a model from a DDP wrapper, if necessary."""
    return model.module if hasattr(model, 'module') else model


def setup_modality_info(args: argparse.Namespace) -> Dict[str, dict]:
    """Sets up the modality info dictionary for the given domains."""
    modality_info = {mod: MODALITY_INFO[mod] for mod in args.all_domains}
    return modality_info


def get_crop_size(crop_coords: torch.Tensor) -> torch.Tensor:
    """Returns the crop heights and widths from the crop coordinates."""
    heights = crop_coords[:,2] - crop_coords[:,0]
    widths = crop_coords[:,3] - crop_coords[:,1]
    return torch.stack([heights, widths], dim=1)


def get_args() -> argparse.Namespace:
    """Parses the arguments from the command line."""
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser('Diffusion VQ-VAE training script', add_help=False)

    # Model parameters
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Patch size for ViT encoder (default: %(default)s)')
    parser.add_argument('--input_size_min', default=224, type=int,
                        help='Minimum image size (default: %(default)s)')
    parser.add_argument('--input_size_max', default=512, type=int,
                        help='Maximum image size (default: %(default)s)')
    parser.add_argument('--resolution_step', default=32, type=int,
                        help='Interval between different training resolutions (default: %(default)s)')
    parser.add_argument('--input_size_enc', default=None, type=int,
                        help='Only used when frozen encoder pos emb are initialized at a certain resolution. (default: %(default)s)')
    parser.add_argument('--encoder_type', default='vit_s_enc', type=str, metavar='ENC',
                        help='Name of encoder (default: %(default)s)')
    parser.add_argument('--decoder_type', default='unet_cat', type=str, metavar='DEC',
                        help='Name of decoder (default: %(default)s)')
    parser.add_argument('--post_mlp', action='store_true')
    parser.add_argument('--no_post_mlp', action='store_false', dest='post_mlp')
    parser.set_defaults(post_mlp=True)
    parser.add_argument('--encoder_ckpt', default=None, type=str,
                        help='Optional path to encoder checkpoint (default: %(default)s)')
    parser.add_argument('--full_ckpt', default=None, type=str,
                        help='Optional path to encoder + quantizer + decoder checkpoint (default: %(default)s)')                 
    parser.add_argument('--freeze_enc', action='store_true',
                        help='Freeze encoder and quantizer (default: %(default)s)')
    parser.add_argument('--no_freeze_enc', action='store_false', dest='freeze_enc')
    parser.set_defaults(freeze_enc=False)
    parser.add_argument('--dec_transformer_dropout', default=0.2, type=int,
                        help='Dropout ratio for the transformer midblock of the UViT decoder (default: %(default)s)')
    
    # Quantizer parameters
    parser.add_argument('--quantizer_type', default='lucid', type=str, metavar='QUANT',
                        help='Type of quantizer. Either lucid or memcodes (default: %(default)s)')
    parser.add_argument('--codebook_size', default=8192, 
                        help="""Size of the VQ code book. For FSQ, this is a string of integers separated by hyphen, 
                             specifying the number levels for each dimension. (default: %(default)s)""")
    parser.add_argument('--latent_dim', default=32, type=int,
                        help='Dimension of the bottleneck. For FSQ, this is set to the number of levels in codebook_size and is ignored. (default: %(default)s)')
    parser.add_argument('--norm_codes', action='store_true')
    parser.add_argument('--no_norm_codes', action='store_false', dest='norm_codes')
    parser.set_defaults(norm_codes=True)
    parser.add_argument('--norm_latents', action='store_true')
    parser.add_argument('--no_norm_latents', action='store_false', dest='norm_latents')
    parser.set_defaults(norm_latents=False)
    parser.add_argument('--codebook_weight', default=1.0, type=float,
                        help='Weight of code book loss (default: %(default)s)')
    parser.add_argument('--quantizer_ema_decay', default=0.8, type=float,
                        help='Quantizer EMA decay rate (default: %(default)s)')
    parser.add_argument('--coef_ema_dead_code', default=4.0, type=float,
                        help='Dead code restart coefficient (default: %(default)s)')
    parser.add_argument('--code_replacement_policy', default='batch_random', type=str,
                        help='Method of replacing dead codes. batch_random or linde_buzo_gray. (default: %(default)s)')
    parser.add_argument('--commitment_weight', default=1.0, type=float,
                        help='Quantizer commitment weight, aka "beta" (default: %(default)s)')
    parser.add_argument('--kmeans_init', action='store_true')
    parser.add_argument('--no_kmeans_init', action='store_false', dest='kmeans_init')
    parser.set_defaults(kmeans_init=False)           

    # Diffusion parameters
    parser.add_argument('--num_train_timesteps', default=1000, type=int,
                        help='Number of diffusion steps during training (default: %(default)s)')
    parser.add_argument('--prediction_type', default='sample', type=str,
                        help='sample (x_0), epsilon (noise), v_prediction (velocity), or v_prediction-epsilon_loss diffusion mode (default: %(default)s)')
    parser.add_argument('--beta_schedule', default='linear', type=str,
                        help='Forward process beta schedule. linear or squaredcos_cap_v2 (default: %(default)s)')
    parser.add_argument('--zero_terminal_snr', action='store_true',
                        help='Enforce SNR of beta schedule to be zero at t=T. (default: %(default)s)')
    parser.add_argument('--no_zero_terminal_snr', action='store_false', dest='zero_terminal_snr')
    parser.set_defaults(zero_terminal_snr=True)
    parser.add_argument('--cls_free_guidance_dropout', default=0.2, type=int,
                        help='Condition dropout percentage during training for classifier free guidance (default: %(default)s)')
    parser.add_argument('--masked_cfg', action='store_true',
                        help='Enable to perform masking on the encoded tokens. (default: %(default)s)')
    parser.add_argument('--no_masked_cfg', action='store_false', dest='masked_cfg')
    parser.set_defaults(masked_cfg=True)
    parser.add_argument('--masked_cfg_low', default=0, type=int,
                        help='Lower bound of number of tokens to mask out (default: %(default)s)')
    parser.add_argument('--masked_cfg_high', default=None, type=int,
                        help='Upper bound of number of tokens to mask out, defaults to total number of tokens minus 1 (default: %(default)s)')
    parser.add_argument('--thresholding', default=True, type=bool,
                        help='Whether or not to dynamically clip outputs to [-1,1]. Only affects inference time. (default: %(default)s)')
    parser.add_argument('--loss_fn', default='mse', type=str,
                        help='Diffusion noise loss function. mse, l1, or smooth_l1 (default: %(default)s)')
    parser.add_argument('--conditioning', default='concat', type=str,
                        help='Method to condition UViT Transformer on tokens. concat or xattn. (default: %(default)s)')
    parser.add_argument('--resolution_cond', action='store_true',
                        help='Enable to condition diffusion decoder on original image resolution. (default: %(default)s)')
    parser.add_argument('--no_resolution_cond', action='store_false', dest='resolution_cond')
    parser.set_defaults(resolution_cond=False)
    parser.add_argument('--eval_res_cond', default=512, type=int,
                        help='"Original" resolution to condition diffusion decoder on during evaluation. (default: %(default)s)')

    # Optimizer parameters
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (default: %(default)s)')
    parser.add_argument('--batch_size_eval', default=None, type=int,
                        help='Batch size per GPU during evaluation (default: %(default)s)')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs (default: %(default)s)')
    parser.add_argument('--save_ckpt_freq', default=10, type=int,
                        help='Checkpoint saving frequency in epochs (default: %(default)s)')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer epsilon (default: %(default)s)')
    parser.add_argument('--opt_betas', default=[0.9, 0.95], type=float, nargs='+', metavar='BETA',
                        help='Optimizer betas (default: %(default)s)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='CLIPNORM',
                        help='Clip gradient norm (default: %(default)s)')
    parser.add_argument('--skip_grad', type=float, default=None, metavar='SKIPNORM',
                        help='Skip update if gradient norm larger than threshold (default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (default: %(default)s)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD.  (Set the same value as args.weight_decay to keep weight decay unchanged)""")

    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256 (default: %(default)s)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='Warmup learning rate (default: %(default)s)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Lower lr bound for cyclic schedulers that hit 0 (default: %(default)s)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='Epochs to warmup LR, if scheduler supports (default: %(default)s)')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='Epochs to warmup LR, if scheduler supports (default: %(default)s)')

    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'bfloat16', 'float32', 'bf16', 'fp16', 'fp32'],
                        help='Data type (default: %(default)s')

    parser.add_argument('--model_ema', action='store_true', default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
    parser.add_argument('--model_ema_update_freq', type=int, default=1, help='')

    # Augmentation parameters
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Probability of horizontal flip (default: %(default)s)')

    # Dataset parameters
    parser.add_argument('--domain', default='rgb', type=str,
                        help='Domain/Task name to load (default: %(default)s)')
    parser.add_argument('--mask_value', default=None, type=float,
                        help='Optionally set masked-out regions to this value after data augs (default: %(default)s)') 
    parser.add_argument('--data_path', default=None, type=str, help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str, help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')
    parser.add_argument('--standardize_surface_normals', default=False, action='store_true')
    parser.add_argument('--min_crop_scale', default=0.8, type=float,
                        help='Minimum crop scale for random data augmentation (default: %(default)s)')
    parser.add_argument('--cache_datasets', default=False, action='store_true',
                        help='Cache file paths in data_path/dataloader_cache for faster Dataset initialization (default: %(default)s)')
    
    parser.add_argument('--use_wds', action='store_true', help='webdatasets')
    parser.add_argument('--no_use_wds', action='store_false', dest='use_wds')
    parser.set_defaults(use_wds=False)
    parser.add_argument('--s3_endpoint', default='', type=str, help='S3 endpoint URL')
    parser.add_argument('--s3_data_endpoint', default=None, type=str, 
                        help='S3 endpoint URL for the data (if different). If set to None, will be set to s3_endpoint')
    parser.add_argument('--wds_n_repeats', default=1, type=int, help='Number of repeats for webdataset loader to improve efficiency')
    parser.add_argument('--wds_shuffle_buffer_tar', default=1_000, type=int, help='Webdatasets shuffle buffer after loading tar files')
    parser.add_argument('--wds_shuffle_buffer_repeat', default=1_000, type=int, help='Webdatasets shuffle buffer after repeating samples')
    parser.add_argument('--s3_multipart_chunksize_mb', default=512, type=int)
    parser.add_argument('--s3_multipart_threshold_mb', default=512, type=int)
    parser.add_argument('--dataset_size', default=None, type=int, help='Needed for DDP when using webdatasets')

    # Eval parameters
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--no_dist_eval', action='store_false', dest='dist_eval',
                    help='Disabling distributed evaluation')
    parser.set_defaults(dist_eval=True)

    parser.add_argument('--step_eval', action='store_true', default=False, help="Evaluate on a step basis")
    parser.add_argument('--epoch_eval', action='store_false', dest='step_eval', help="Evaluate on an epoch basis")
    parser.add_argument('--eval_noise_schedule', default='DDIMScheduler', type=str,
                        help='Type of diffusers.schedulers noise scheduler for evaluation. (default: %(default)s)')
    parser.add_argument('--num_eval_timesteps', default=50, type=int,
                        help='Number of diffusion steps during evaluation (default: %(default)s)')
    parser.add_argument('--input_size_eval', default="256", type=str,
                        help='Evaluation is ran at this list of image sizes, split by  a hyphen (+ min and max size if they are different) (default: %(default)s)')
    parser.add_argument('--num_eval_metrics_samples', default=None, type=int,
                        help='Number of samples to use for computing evaluation metrics (default: %(default)s)')
    parser.add_argument('--eval_freq', default=1, type=int, help="frequency of evaluation (in iterations or epochs)")
    parser.add_argument('--eval_metrics_freq', default=1, type=int, help="frequency of evaluation metrics (in iterations or epochs)")
    parser.add_argument('--eval_image_log_freq', default=5, type=int, help="frequency of evaluation image logging (in iterations)")
    parser.add_argument('--num_logged_images', default=100, type=int, help="number of images to log")
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--no_inception', action='store_true', default=False, help="Disable Inception metric during eval")
    parser.add_argument('--log_codebook_usage', action='store_true', help='Log the codebook usage')
    parser.add_argument('--no_codebook_usage', action='store_false', dest='log_codebook_usage', help='Disable logging of the codebook usage')
    parser.set_defaults(log_codebook_usage=True)


    # Misc.
    parser.add_argument('--output_dir', default='',
                        help='Path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training / testing')

    parser.add_argument('--seed', default=0, type=int, help='Random seed ')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--no_find_unused_params', action='store_false', dest='find_unused_params')
    parser.set_defaults(find_unused_params=False)

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
    parser.add_argument('--wandb_tags', default='', type=str, help='Extra wandb tags, separated by a double hyphen')
    parser.add_argument('--show_user_warnings', default=False, action='store_true')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # S3 Load & Save
    parser.add_argument('--s3_path', default='', type=str, help='S3 path to model')
    parser.add_argument('--s3_save_dir', type=str, default="")

    # Parse config file if there is one
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Add the config path as a final args if given
    args.config_path = args_config.config

    return args


def get_model(args: argparse.Namespace, device: Union[torch.device, str]) -> DiVAE:
    """Creates and returns model from arguments."""

    # Compute the dead codebook threshold
    total_batch_size = args.batch_size * utils.get_world_size()
    mean_img_size = (args.input_size_min + args.input_size_max) // 2
    tokens_per_image = (mean_img_size // args.patch_size) ** 2
    codebook_size_int = np.prod([int(d) for d in args.codebook_size.split('-')]) if isinstance(args.codebook_size, str) else args.codebook_size
    uniform_token_count_per_batch = total_batch_size * tokens_per_image / codebook_size_int
    threshold_ema_dead_code = uniform_token_count_per_batch / args.coef_ema_dead_code
    print(f'Computed dead code EMA threshold: {threshold_ema_dead_code:.4f}')

    ignore_keys = ['decoder', 'loss', 'post_quant_conv', 'post_quant_proj', 'encoder.pos_emb']
    ckpt = args.encoder_ckpt
    if args.full_ckpt is not None:
        ignore_keys = ['encoder.pos_emb']
        ckpt = args.full_ckpt

    n_channels = MODALITY_INFO[args.domain]['num_channels']
    if args.mask_value is not None:
        n_channels += 1

    model = DiVAE(
        image_size=args.input_size_max,
        image_size_enc=args.input_size_enc,
        n_channels=n_channels,
        enc_type=args.encoder_type,
        dec_type=args.decoder_type,
        post_mlp=args.post_mlp,
        quant_type=args.quantizer_type,
        patch_size=args.patch_size,
        codebook_size=args.codebook_size,
        latent_dim=args.latent_dim,
        norm_codes=args.norm_codes,
        norm_latents=args.norm_latents,
        prediction_type=args.prediction_type.split('-')[0],
        num_train_timesteps=args.num_train_timesteps,
        ckpt_path=ckpt,
        ignore_keys=ignore_keys,
        freeze_enc=args.freeze_enc,
        cls_free_guidance_dropout=args.cls_free_guidance_dropout,
        masked_cfg=args.masked_cfg,
        masked_cfg_low=args.masked_cfg_low,
        masked_cfg_high=args.masked_cfg_high,
        beta_schedule=args.beta_schedule,
        thresholding=args.thresholding,
        sync_codebook=True,
        ema_decay=args.quantizer_ema_decay,
        threshold_ema_dead_code=threshold_ema_dead_code,
        code_replacement_policy=args.code_replacement_policy,
        commitment_weight=args.commitment_weight,
        kmeans_init=args.kmeans_init,
        undo_std=False,
        conditioning=args.conditioning,
        dec_transformer_dropout=args.dec_transformer_dropout,
        zero_terminal_snr=args.zero_terminal_snr,
    )

    return model.to(device)


def main(args: argparse.Namespace) -> None:
    """Main function for training and evaluation."""
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

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

    num_tasks = utils.get_world_size()
    args.num_tasks = num_tasks
    global_rank = utils.get_rank()
    sampler_rank = global_rank

    args.eval_res_cond = to_2tuple(args.eval_res_cond) if args.resolution_cond else None

    args.all_domains = [args.domain] if args.mask_value is None else [args.domain, 'mask_valid']
    
    modality_info = setup_modality_info(args)
    modality_paths = {mod: modality_info[mod]['path'] for mod in modality_info if modality_info[mod].get('path', None) is not None}

    args.input_size = args.input_size_max # For multi-resolution training, load the largest resolution and downsample accordingly
    image_augmenter_train = RandomCropImageAugmenter(target_size=args.input_size, main_domain=args.domain, crop_scale=(args.min_crop_scale, 1.0))

    MODALITY_TRANSFORMS_DIVAE['normal'] = NormalTransform(standardize_surface_normals=args.standardize_surface_normals)
    MODALITY_TRANSFORMS_DIVAE['rgb'] = RGBTransform(imagenet_default_mean_and_std=args.imagenet_default_mean_and_std)

    if args.use_wds:
        if args.data_path.startswith("s3"):
            # When loading from S3 using boto3, hijack webdatasets tar loading
            MB = 1024 ** 2
            transfer_config = TransferConfig(
                multipart_threshold=args.s3_multipart_threshold_mb * MB, 
                multipart_chunksize=args.s3_multipart_chunksize_mb * MB, 
                max_io_queue=1000)

            s3_client = boto3.client(
                service_name='s3',
                aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                endpoint_url=args.s3_data_endpoint,
            )

            def get_bytes_io(path):
                byte_io = io.BytesIO()
                _, bucket, key, _ = re.split("s3://(.*?)/(.*)$", path)
                s3_client.download_fileobj(bucket, key, byte_io, Config=transfer_config)
                byte_io.seek(0)
                return byte_io

            def url_opener(data, handler=reraise_exception, **kw):
                for sample in data:
                    url = sample["url"]
                    try:
                        stream = get_bytes_io(url)
                        sample.update(stream=stream)
                        yield sample
                    except Exception as exn:
                        exn.args = exn.args + (url,)
                        if handler(exn):
                            continue
                        else:
                            break

            wds.tariterators.url_opener = url_opener

        # When using webdatasets
        data_loader_train = build_wds_divae_dataloader(
            data_path=args.data_path, modality_info=modality_info, modality_transforms=MODALITY_TRANSFORMS_DIVAE,
            image_augmenter=image_augmenter_train, num_gpus=num_tasks, num_workers=args.num_workers,
            batch_size=args.batch_size, epoch_size=args.dataset_size, shuffle_buffer_load=args.wds_shuffle_buffer_tar,
            shuffle_buffer_repeat=args.wds_shuffle_buffer_repeat, n_repeats=args.wds_n_repeats,
        )

        
        num_training_steps_per_epoch = args.dataset_size // (args.batch_size * num_tasks)
    else:
        transforms_train = UnifiedDataTransform(transforms_dict=MODALITY_TRANSFORMS_DIVAE, image_augmenter=image_augmenter_train, add_sizes=args.resolution_cond)
        dataset_train = MultiModalDatasetFolder(root=args.data_path, modalities=args.all_domains, modality_paths=modality_paths, 
                                                modality_transforms=MODALITY_TRANSFORMS_DIVAE, transform=transforms_train, cache=args.cache_datasets)

        num_training_steps_per_epoch = len(dataset_train) // (args.batch_size * num_tasks)
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True, drop_last=True,
        )
        print("Sampler_train = %s" % str(sampler_train))

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

    if args.eval_data_path:
        image_augmenter_val = CenterCropImageAugmenter(target_size=args.input_size, main_domain=args.domain)
        transforms_val = UnifiedDataTransform(transforms_dict=MODALITY_TRANSFORMS_DIVAE, image_augmenter=image_augmenter_val, add_sizes=args.resolution_cond)

        dataset_val = MultiModalDatasetFolder(root=args.eval_data_path, modalities=args.all_domains, modality_paths=modality_paths, 
                                              modality_transforms=MODALITY_TRANSFORMS_DIVAE, transform=transforms_val, cache=args.cache_datasets)
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size) if args.batch_size_eval is None else args.batch_size_eval,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        # Computing image metrics can be expensive (because of the many diffusion forward passes), so we can choose to only do it on a subset of the data
        if args.num_eval_metrics_samples is not None:
            dataset_metrics = MultiModalDatasetFolder(root=args.eval_data_path, modalities=args.all_domains, modality_paths=modality_paths, 
                                                    modality_transforms=MODALITY_TRANSFORMS_DIVAE, transform=transforms_val, 
                                                    pre_shuffle=True, max_samples=args.num_eval_metrics_samples, cache=args.cache_datasets)
            if args.dist_eval:
                if len(dataset_metrics) % num_tasks != 0:
                    print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
                sampler_metrics = torch.utils.data.DistributedSampler(
                    dataset_metrics, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            else:
                sampler_metrics = torch.utils.data.SequentialSampler(dataset_metrics)

            data_loader_metrics = torch.utils.data.DataLoader(
                dataset_metrics, sampler=sampler_metrics,
                batch_size=int(1.5 * args.batch_size) if args.batch_size_eval is None else args.batch_size_eval,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )
        else:
            data_loader_metrics = data_loader_val

        if args.num_logged_images is not None:
            dataset_image_log = MultiModalDatasetFolder(root=args.eval_data_path, modalities=args.all_domains, modality_paths=modality_paths, 
                                                    modality_transforms=MODALITY_TRANSFORMS_DIVAE, transform=transforms_val, 
                                                    pre_shuffle=True, max_samples=args.num_logged_images, cache=args.cache_datasets)
            # No dist eval, we only run it on the main process
            sampler_image_log = torch.utils.data.SequentialSampler(dataset_image_log)
            data_loader_image_log = torch.utils.data.DataLoader(
                dataset_image_log, sampler=sampler_image_log,
                batch_size=int(1.5 * args.batch_size) if args.batch_size_eval is None else args.batch_size_eval,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )
        else:
            data_loader_image_log = data_loader_val

    else:
        data_loader_val, data_loader_metrics, dataset_image_log = None, None, None

    if global_rank == 0 and args.log_wandb:
        log_writer = utils.WandbLogger(args)
        log_writer.set_step(0)
    else:
        log_writer = None

    if global_rank == 0 and args.log_wandb:
        # Edit run name and add tags
        args.wandb_tags = args.wandb_tags.split('--') if args.wandb_tags else []
        log_writer = utils.WandbLogger(args)
        log_writer.set_step(0)
    else:
        log_writer = None

    print(args)

    model = get_model(args, device)
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)
    else:
        model_ema = None
    
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model = %s" % str(model_without_ddp))
    print(f"Number of params: {n_parameters / 1e6} M")

    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.blr * total_batch_size / 256

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler(enabled=dtype == torch.float16)

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, 
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
    
    # Evaluation noise scheduler
    if args.eval_noise_schedule in ['DDPMScheduler', 'DDIMScheduler']:
        eval_noise_schedule = getattr(sys.modules[__name__], args.eval_noise_schedule)(
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule=args.beta_schedule,
            prediction_type=args.prediction_type.split('-')[0],
            thresholding=args.thresholding,
            clip_sample=False,
            zero_terminal_snr=args.zero_terminal_snr
        )
    elif args.eval_noise_schedule is not None:
        eval_noise_schedule = getattr(diffusers_schedulers, args.eval_noise_schedule)(
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule=args.beta_schedule,
            prediction_type=args.prediction_type.split('-')[0],
            thresholding=args.thresholding,
            clip_sample=False
        )
    else:
        eval_noise_schedule = None

    # The various train resolutions
    train_res_choices = list(range(args.input_size_min, args.input_size_max+args.resolution_step, args.resolution_step))

    if isinstance(args.input_size_eval, str):
        args.input_size_eval = [int(s) for s in args.input_size_eval.split("-")]
    elif isinstance(args.input_size_eval, int):
        args.input_size_eval = [args.input_size_eval]
    eval_image_sizes = set([*args.input_size_eval, train_res_choices[0], train_res_choices[-1]])

    if args.eval_only:
        # Evaluate the model
        eval_stats = evaluate(
            model, data_loader_val, device, args.domain, train_res_choices, args.prediction_type, 
            args.loss_fn, args.codebook_weight, dtype=dtype, mask_value=args.mask_value,
        )
        if log_writer is not None:
            log_writer.update(eval_stats)

        # Evaluate several common metrics at eval resolution, min train resolution and max train resolutions
        for eval_img_size in eval_image_sizes:
            eval_metrics_results = eval_metrics(
                model, data_loader_metrics, device, args.domain, eval_img_size, eval_noise_schedule, 
                args.num_eval_timesteps, dtype=dtype, mask_value=args.mask_value,
                no_inception=args.no_inception, log_writer=log_writer, log_codebook_usage=args.log_codebook_usage,
            )
            if log_writer is not None:
                log_writer.update(eval_metrics_results)

            eval_image_log(model, data_loader_image_log, device, args.domain, eval_img_size, eval_noise_schedule, 
                           args.num_eval_timesteps, dtype=dtype, num_logged_images=args.num_logged_images, 
                           mask_value=args.mask_value, log_writer=log_writer)

        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            domain=args.domain,
            codebook_weight=args.codebook_weight,
            train_res_choices=train_res_choices, 
            eval_image_sizes=eval_image_sizes,
            eval_noise_schedule=eval_noise_schedule,
            num_eval_timesteps=args.num_eval_timesteps,
            model_ema=model_ema,
            max_norm=args.clip_grad,
            max_skip_norm=args.skip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            dtype=dtype,
            loader_len=num_training_steps_per_epoch,
            data_loader_val=data_loader_val, 
            data_loader_metrics=data_loader_metrics,
            data_loader_image_log=data_loader_image_log,
            eval_freq=args.eval_freq,
            eval_metrics_freq=args.eval_metrics_freq,
            eval_image_log_freq=args.eval_image_log_freq,
            num_logged_images=args.num_logged_images,
            prediction_type=args.prediction_type,
            loss_fn=args.loss_fn,
            ema_freq=args.model_ema_update_freq,
            mask_value=args.mask_value,
            no_inception=args.no_inception,
            log_codebook_usage=args.log_codebook_usage,
            step_eval=args.step_eval,
        )

        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
                if epoch + 1 == args.epochs:
                        use_s3 = len(args.s3_save_dir) > 0
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, ckpt_name='final', use_s3=use_s3)

        log_stats = {**{k: v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        # Evaluation (if we evaluate on an epoch-basis)
        if not args.step_eval or (epoch + 1 == args.epochs):
            launch_evaluate = (data_loader_val is not None) and ((epoch % args.eval_freq == 0) or (epoch + 1 == args.epochs))
            launch_eval_metrics = (data_loader_metrics is not None) and args.eval_metrics_freq > 0 and  ((epoch % args.eval_metrics_freq == 0) or (epoch + 1 == args.epochs))
            launch_eval_image_log = (data_loader_image_log is not None) and args.eval_image_log_freq > 0 and  ((epoch % args.eval_image_log_freq == 0) or (epoch + 1 == args.epochs))

            eval_stats = launch_evals(
                launch_evaluate=launch_evaluate, launch_eval_metrics=launch_eval_metrics, launch_eval_image_log=launch_eval_image_log,
                model=model, device=device, domain=args.domain, codebook_weight=args.codebook_weight, train_res_choices=train_res_choices,
                eval_image_sizes=eval_image_sizes, eval_noise_schedule=eval_noise_schedule, num_eval_timesteps=args.num_eval_timesteps,
                model_ema=model_ema, dtype=dtype, data_loader_val=data_loader_val, data_loader_metrics=data_loader_metrics,
                data_loader_image_log=data_loader_image_log, num_logged_images=args.num_logged_images, prediction_type=args.prediction_type,
                loss_fn=args.loss_fn, mask_value=args.mask_value, no_inception=args.no_inception, 
                log_writer=log_writer, log_codebook_usage=args.log_codebook_usage,
            )
            
            log_stats.update(eval_stats)

        if log_writer is not None:
            log_writer.update(log_stats)

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def mask_out_samples(clean_inputs: torch.Tensor, 
                     mask_valid: Optional[torch.BoolTensor] = None, 
                     mask_value: Optional[float] = None) -> torch.Tensor:
    """Optionally mask out invalid regions and concat mask to images.
    Useful when tokenizing simulated data that contains unlabeled regions.

    Args:
        clean_inputs: Input images
        mask_valid: Boolean mask of valid regions. 
          True = keep, False = replace by mask_value
        mask_value: Value to replace invalid regions with

    Returns:
        Masked out images with C+1 channels (mask is the last channel).
        The mask is converted to [-1, 1] range, where -1 is invalid and 1 is valid.
    """
    if mask_valid is not None and mask_value is not None:
        mask_valid = mask_valid.to(clean_inputs.device, non_blocking=True)
        clean_inputs[~repeat(mask_valid, 'b 1 h w -> b n h w', n=clean_inputs.shape[1])] = mask_value
        mask_valid = mask_valid.float() * 2 - 1 # Valid regions -> 1, Masked-out regions -> -1
        clean_inputs = torch.cat([clean_inputs, mask_valid], dim=1)
    return clean_inputs

def train_one_epoch(model: Union[nn.Module, DDP], 
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, 
                    device: Union[torch.device, str], 
                    epoch: int, 
                    loss_scaler: NativeScaler, 
                    domain: str, 
                    codebook_weight: float, 
                    train_res_choices: List[int], 
                    eval_image_sizes: Set[int], 
                    eval_noise_schedule: SchedulerMixin,
                    num_eval_timesteps: int,
                    model_ema: Optional[ModelEma] = None, 
                    max_norm: Optional[float] = None, 
                    max_skip_norm: Optional[float] = None, 
                    log_writer: Optional[utils.WandbLogger] = None, 
                    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None, 
                    start_steps: int = None, 
                    lr_schedule_values: Iterable[float] = None, 
                    wd_schedule_values: Iterable[float] = None, 
                    dtype: torch.dtype = torch.float16, 
                    loader_len: Optional[int] = None, 
                    data_loader_val: Optional[Iterable] = None, 
                    data_loader_metrics: Optional[Iterable] = None, 
                    data_loader_image_log: Optional[Iterable] = None,
                    eval_freq: int = 1000, 
                    eval_metrics_freq: int = 10_000, 
                    eval_image_log_freq: int = 10_000, 
                    num_logged_images: int = 100,
                    prediction_type: str = 'v_prediction',
                    loss_fn: str = 'mse', 
                    ema_freq: int = 1, 
                    mask_value: Optional[float] = None, 
                    no_inception: bool = False, 
                    log_codebook_usage: bool = True, 
                    step_eval: bool = False) -> Dict[str, float]:
    """Perform one training epoch and return stats. The image resolution is 
    randomly sampled from train_res_choices. At the specified intervals,
    evaluation is performed, metrics are computed, and images are logged.

    Args:
        model: Model to train.
        data_loader: Training data loader.
        optimizer: Optimizer.
        device: Device to train on.
        epoch: Epoch number.
        loss_scaler: Loss scaler for mixed precision.
        domain: Image domain.
        codebook_weight: Codebook loss weight.
        train_res_choices: List of training resolutions to randomly sample from.
        eval_image_sizes: [Eval] Set of evaluation resolutions to perform eval on.
        eval_noise_schedule: Noise schedule to use for diffusion.
        num_eval_timesteps: Number of diffusion timesteps to use for evaluation.
        model_ema: Optional EMA model to update every ema_freq iterations.
        max_norm: Max norm for gradient clipping.
        max_skip_norm: Max norm for gradient skipping.
        log_writer: Optional wandb logger.
        lr_scheduler: Optional learning rate scheduler.
        start_steps: Epoch start steps to compute global training iteration.
        lr_schedule_values: Learning rate schedule values.
        wd_schedule_values: Weight decay schedule values.
        dtype: Data type for mixed precision training.
        loader_len: Length of the data loader.
        data_loader_val: [Eval] Dataloader for standard evaluation.
        data_loader_metrics: [Eval] Dataloader for evaluation of image metrics.
        data_loader_image_log: [Eval] Dataloader for image logging.
        eval_freq: [Eval] Frequency at which to perform standard evaluation.
        eval_metrics_freq: [Eval] Frequency at which to compute image metrics.
        eval_image_log_freq: [Eval] Frequency at which to log images.
        num_logged_images: [Eval] Number of images to log.
        prediction_type: Type of diffusion target.
        loss_fn: Reconstruction loss function identifyer.
        ema_freq: Frequency at which to update the EMA model.
        mask_value: Value to mask out invalid regions with.
        no_inception: [Eval] Whether to skip Inception score computation.
        log_codebook_usage: [Eval] Whether to compute and log codebook usage.
        step_eval: [Eval] Whether to perform evaluation on a step-basis instead of epoch-basis.

    Returns:
        Training stats.
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for step, x in enumerate(metric_logger.log_every(data_loader, print_freq, iter_len=loader_len, header=header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration

        # Evaluation (if we evaluate on a step-basis)
        if step_eval:
            launch_evaluate = (data_loader_val is not None) and (it % eval_freq == 0) and (it != 0)
            launch_eval_metrics = (data_loader_metrics is not None) and (it % eval_metrics_freq == 0) and (it != 0)
            launch_eval_image_log = (data_loader_image_log is not None) and (it % eval_image_log_freq == 0) and (it != 0)

            eval_stats = launch_evals(
                launch_evaluate=launch_evaluate, launch_eval_metrics=launch_eval_metrics, launch_eval_image_log=launch_eval_image_log,
                model=model, device=device, domain=domain, codebook_weight=codebook_weight, train_res_choices=train_res_choices,
                eval_image_sizes=eval_image_sizes, eval_noise_schedule=eval_noise_schedule, num_eval_timesteps=num_eval_timesteps,
                model_ema=model_ema, dtype=dtype, data_loader_val=data_loader_val, data_loader_metrics=data_loader_metrics,
                data_loader_image_log=data_loader_image_log, num_logged_images=num_logged_images, prediction_type=prediction_type,
                loss_fn=loss_fn, mask_value=mask_value, no_inception=no_inception, log_writer=log_writer, log_codebook_usage=log_codebook_usage, 
                )
            
            if log_writer is not None:
                log_writer.update(eval_stats)

        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # Prepare clean and noised images
        clean_images = x[domain].to(device, non_blocking=True)

        # Optionally mask out invalid regions and concat mask and images
        clean_images = mask_out_samples(clean_images, x.get('mask_valid', None), mask_value=mask_value)
        
        # Randomly sample an image size between the min and max for this batch and resize the images
        res_idx = hash(str(it)) % len(train_res_choices)
        image_size = train_res_choices[res_idx]
        clean_images = F.interpolate(clean_images, image_size, mode='bilinear', align_corners=False)

        # Sample noise that we'll add to the images
        noise = torch.randn(clean_images.shape).to(device)
        # Sample a uniformly random timestep for each image
        timesteps = torch.randint(
            0, unwrap_model(model).noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],)
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = unwrap_model(model).noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Optionally condition diffusion model on original resolution
        orig_res = get_crop_size(x['crop_coords']).to(device) if 'crop_coords' in x else None

        with torch.cuda.amp.autocast(dtype=dtype, enabled=dtype != torch.float32):
            model_output, code_loss = model(clean_images, noisy_images, timesteps.to(device), orig_res=orig_res)

            if prediction_type == 'sample':
                target = clean_images
            elif prediction_type == 'epsilon':
                target = noise
            elif prediction_type == 'v_prediction':
                target = unwrap_model(model).noise_scheduler.get_velocity(clean_images, noise, timesteps)
            elif prediction_type == 'v_prediction-epsilon_loss':
                target = noise
                model_output = unwrap_model(model).noise_scheduler.get_noise(noisy_images, model_output, timesteps)
            
            if loss_fn == 'mse':
                reconst_loss = F.mse_loss(model_output, target)
            elif loss_fn == 'l1':
                reconst_loss = F.l1_loss(model_output, target)
            elif loss_fn == 'smooth_l1':
                reconst_loss = F.smooth_l1_loss(model_output, target)

        loss = reconst_loss + codebook_weight * code_loss
        
        loss_value = loss.item()
        reconst_loss_value = reconst_loss.item()
        code_loss_value = code_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)


        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm, skip_grad=max_skip_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        optimizer.zero_grad()

        if model_ema is not None and it % ema_freq == 0:
            # EMA every few iterations
            model_ema.update(model)

        if dtype == torch.float16:
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(reconst_loss=reconst_loss_value)
        metric_logger.update(code_loss=code_loss_value)
        if dtype == torch.float16:
            metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(
                {
                    'loss': loss_value,
                    'reconst_loss': reconst_loss_value,
                    'code_loss': code_loss_value,
                    'lr': max_lr,
                    'weight_decay': weight_decay_value,
                    'grad_norm': grad_norm,
                }
            )
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {'[Epoch] ' + k: meter.global_avg for k, meter in metric_logger.meters.items()}

def launch_evals(
        launch_evaluate: bool, 
        launch_eval_metrics: bool, 
        launch_eval_image_log: bool,
        model: Union[nn.Module, DDP], 
        device: Union[torch.device, str], 
        domain: str, 
        codebook_weight: float, 
        train_res_choices: List[int], 
        eval_image_sizes: Set[int], 
        eval_noise_schedule: SchedulerMixin,
        num_eval_timesteps: int,
        model_ema: Optional[ModelEma] = None, 
        dtype: torch.dtype = torch.float16, 
        data_loader_val: Optional[Iterable] = None, 
        data_loader_metrics: Optional[Iterable] = None, 
        data_loader_image_log: Optional[Iterable] = None, 
        num_logged_images: int = 100, 
        prediction_type: str = 'v_prediction',
        loss_fn: str = 'mse', 
        mask_value: Optional[float] = None, 
        eval_res_cond: Optional[int] = None,
        no_inception: bool = False,
        log_writer: Optional[utils.WandbLogger] = None,
        log_codebook_usage: bool = True) -> Dict[str, float]:
    """Launcher for various evaluation functions: standard evaluation,
    evaluation of image metrics, and image logging.
    
    Args:
        launch_evaluate: Whether to launch standard evaluation.
        launch_eval_metrics: Whether to launch evaluation of image metrics.
        launch_eval_image_log: Whether to launch image logging.
        model: Model to evaluate.
        device: Device to evaluate on.
        domain: Image domain.
        codebook_weight: Codebook loss weight.
        train_res_choices: List of training resolutions to randomly sample from.
        eval_image_sizes: Set of evaluation resolutions to perform eval on.
        eval_noise_schedule: Noise schedule to use for diffusion.
        num_eval_timesteps: Number of diffusion timesteps to use for evaluation.
        model_ema: Optional EMA model to evaluate for each of the eval functions.
        dtype: Data type for mixed precision inference.
        data_loader_val: Dataloader for standard evaluation.
        data_loader_metrics: Dataloader for evaluation of image metrics.
        data_loader_image_log: Dataloader for image logging.
        num_logged_images: Number of images to log.
        prediction_type: Type of diffusion target.
        loss_fn: Reconstruction loss function identifyer.
        mask_value: Value to mask out invalid regions with.
        eval_res_cond: A fixed eval image size to condition the diffusion on. 
          Ignored if None. See SDXL https://arxiv.org/abs/2307.01952 for more details.
        no_inception: Whether to skip Inception score computation.
        log_writer: Optional wandb logger.
        log_codebook_usage: Whether to compute and log codebook usage.
        
    Returns:
        Dictionary of all evaluation results.
    """
    all_eval_stats = {}
      
    if launch_evaluate:
        eval_stats = evaluate(
            model, data_loader_val, device, domain, train_res_choices, prediction_type, 
            loss_fn, codebook_weight, dtype=dtype, mask_value=mask_value, prefix='[Eval]')
        
        all_eval_stats.update(eval_stats)

        model.train()

        if model_ema is not None:
            eval_stats = evaluate(
                unwrap_model(model_ema), data_loader_val, device, domain, train_res_choices, prediction_type, 
                loss_fn, codebook_weight, dtype=dtype, mask_value=mask_value, prefix='[EMA Eval]')
            all_eval_stats.update(eval_stats)
        
        torch.cuda.empty_cache()


    # Evaluate image metrics and log images
    if launch_eval_metrics:
        # Evaluate several common metrics at eval resolution, min train resolution and max train resolutions
        for eval_img_size in eval_image_sizes:
            eval_metrics_results = eval_metrics(
                model, data_loader_metrics, device, domain, eval_img_size, eval_noise_schedule, 
                num_eval_timesteps, dtype=dtype, mask_value=mask_value, eval_res_cond=eval_res_cond, 
                no_inception=no_inception, prefix='[Eval]', log_writer=log_writer, log_codebook_usage=log_codebook_usage,
            )
            all_eval_stats.update(eval_metrics_results)

        model.train()

        if model_ema is not None:
             # Evaluate several common metrics at eval resolution, min train resolution and max train resolutions
            for eval_img_size in eval_image_sizes:
                eval_metrics_results = eval_metrics(
                    unwrap_model(model_ema), data_loader_metrics, device, domain, eval_img_size, eval_noise_schedule, 
                    num_eval_timesteps, dtype=dtype, mask_value=mask_value, eval_res_cond=eval_res_cond, 
                    no_inception=no_inception, prefix='[EMA Eval]', log_writer=log_writer, log_codebook_usage=log_codebook_usage,
                )
                all_eval_stats.update(eval_metrics_results)

        torch.cuda.empty_cache()


    if launch_eval_image_log:
        # Evaluate several common metrics at eval resolution, min train resolution and max train resolutions
        for eval_img_size in eval_image_sizes:
            eval_image_log(model, data_loader_image_log, device, domain, eval_img_size, eval_noise_schedule, 
                num_eval_timesteps, dtype=dtype, num_logged_images=num_logged_images, mask_value=mask_value, 
                eval_res_cond=eval_res_cond, log_writer=log_writer, prefix='[Eval]')

        model.train()

        if model_ema is not None:
            for eval_img_size in eval_image_sizes:
                eval_image_log(unwrap_model(model_ema), data_loader_image_log, device, domain, eval_img_size, eval_noise_schedule, 
                    num_eval_timesteps, dtype=dtype, num_logged_images=num_logged_images, mask_value=mask_value, 
                    eval_res_cond=eval_res_cond, log_writer=log_writer, prefix='[EMA Eval]')


        torch.cuda.empty_cache()

    return all_eval_stats


@torch.no_grad()
def evaluate(model: Union[nn.Module, DDP], 
             data_loader: Iterable, 
             device: Union[torch.device, str], 
             domain: str, 
             train_res_choices: List[int], 
             prediction_type: str,
             loss_fn: str, 
             codebook_weight: float, 
             dtype: torch.dtype = torch.float16, 
             mask_value: Optional[float] = None, 
             prefix: str = '[Eval]') -> Dict[str, float]:
    """Perform one evaluation epoch and return stats. As during 
    training, the resolution is randomly sampled from train_res_choices.

    Args:
        model: Model to evaluate.
        data_loader: Validation set data loader.
        device: Device to evaluate on.
        domain: Image domain.
        train_res_choices: List of training resolutions to randomly sample from.
        prediction_type: Type of diffusion target.
        loss_fn: Reconstruction loss function identifyer.
        codebook_weight: Codebook loss weight.
        dtype: Data type for mixed precision inference.
        mask_value: Value to mask out invalid regions with.
        prefix: Prefix for wandb logging.

    Returns:
        Dictionary of evaluation results.
    """
    # switch to evaluation mode
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    for step, x in enumerate(metric_logger.log_every(data_loader, print_freq=10, header=f'{prefix} ')):

        # Prepare clean and noised images
        clean_images = x[domain].to(device, non_blocking=True)

        # Optionally mask out invalid regions and concat mask and images
        clean_images = mask_out_samples(clean_images, x.get('mask_valid', None), mask_value=mask_value)
        
        # Randomly sample an image size between the min and max for this batch and resize the images
        res_idx = hash(str(step)) % len(train_res_choices)
        image_size = train_res_choices[res_idx]
        clean_images = F.interpolate(clean_images, image_size, mode='bilinear', align_corners=False)

        # Sample noise that we'll add to the images
        noise = torch.randn(clean_images.shape).to(device)
        # Sample a uniformly random timestep for each image
        timesteps = torch.randint(
            0, unwrap_model(model).noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],)
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = unwrap_model(model).noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Optionally condition diffusion model on original resolution
        orig_res = get_crop_size(x['crop_coords']).to(device) if 'crop_coords' in x else None

        with torch.cuda.amp.autocast(dtype=dtype, enabled=dtype != torch.float32):
            model_output, code_loss = model(clean_images, noisy_images, timesteps.to(device), orig_res=orig_res)

            if prediction_type == 'sample':
                target = clean_images
            elif prediction_type == 'epsilon':
                target = noise
            elif prediction_type == 'v_prediction':
                target = unwrap_model(model).noise_scheduler.get_velocity(clean_images, noise, timesteps)
            elif prediction_type == 'v_prediction-epsilon_loss':
                target = noise
                model_output = unwrap_model(model).noise_scheduler.get_noise(noisy_images, model_output, timesteps)
            
            if loss_fn == 'mse':
                reconst_loss = F.mse_loss(model_output, target)
            elif loss_fn == 'l1':
                reconst_loss = F.l1_loss(model_output, target)
            elif loss_fn == 'smooth_l1':
                reconst_loss = F.smooth_l1_loss(model_output, target)

        loss = reconst_loss + codebook_weight * code_loss
        
        loss_value = loss.item()
        reconst_loss_value = reconst_loss.item()
        code_loss_value = code_loss.item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(reconst_loss=reconst_loss_value)
        metric_logger.update(code_loss=code_loss_value)

    # gather the stats from all processes
    print("Eval averaged stats:", metric_logger)
    return {f'{prefix} {k}': meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval_metrics(model: Union[nn.Module, DDP], 
                 data_loader: Iterable, 
                 device: Union[torch.device, str], 
                 domain: str, 
                 eval_size: int, 
                 noise_schedule: SchedulerMixin,
                 num_diffusion_steps: int,
                 dtype: torch.dtype = torch.float16, 
                 mask_value: Optional[float] = None, 
                 eval_res_cond: Optional[int] = None,
                 no_inception: bool = False, 
                 compute_on_cpu: bool = False, 
                 prefix: str = '[Eval]', 
                 log_writer: Optional[utils.WandbLogger] = None,
                 log_codebook_usage: bool = True) -> Dict[str, float]:
    """Compute validation image metrics (FID, LPIPS, Inception, MS-SSIM, PSNR, MSE)
    using torchmetrics and compute codebook usage stats.

    Args:
        model: Model to evaluate.
        data_loader: Validation set data loader.
        device: Device to evaluate on.
        domain: Image domain.
        eval_size: Resolution to evaluate at.
        noise_schedule: Noise schedule to use for diffusion.
        num_diffusion_steps: Number of diffusion steps to use for eval.
        dtype: Data type for mixed precision inference.
        mask_value: Value to mask out invalid regions with.
        eval_res_cond: A fixed eval image size to condition the diffusion on. 
          Ignored if None. See SDXL https://arxiv.org/abs/2307.01952 for more details.
        no_inception: Whether to skip Inception score computation.
        compute_on_cpu: Whether to compute torchmetrics on CPU.
        prefix: Prefix for wandb logging.
        log_writer: Optional wandb logger.
        log_codebook_usage: Whether to compute and log codebook usage.

    Returns:
        Dictionary of image metrics evaluation results.
    """
    model.eval()

    # Initialize metrics
    mse_metric = MeanSquaredError(squared=True, sync_on_compute=True, compute_on_cpu=compute_on_cpu).to(device)
    mae_metric = MeanAbsoluteError(sync_on_compute=True, compute_on_cpu=compute_on_cpu).to(device)
    psnr_metric = PeakSignalNoiseRatio(
        data_range=1., reduction='elementwise_mean', sync_on_compute=True, compute_on_cpu=compute_on_cpu,
    ).to(device)
    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=1., reduction='elementwise_mean', normalize=False, sync_on_compute=True, compute_on_cpu=compute_on_cpu,
    ).to(device)
    if domain in ['rgb', 'normal']:
        # All of these metrics expect images in [0, 1]
        fid_metric = FrechetInceptionDistance(
            feature=2048, reset_real_features=True, normalize=True, sync_on_compute=True, compute_on_cpu=compute_on_cpu,
        ).to(device)
        lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type='alex', reduction='mean', normalize=True, sync_on_compute=True, compute_on_cpu=compute_on_cpu,
        ).to(device)
        inception_metric = None if no_inception else InceptionScore(
            feature='logits_unbiased', splits=10, normalize=True, sync_on_compute=True, compute_on_cpu=compute_on_cpu,
        ).to(device)
    else:
        fid_metric, lpips_metric, inception_metric = None, None, None
    local_tokens = [] if log_codebook_usage else None # Collects the encoded tokens for all images evaluated on each device

    metric_logger = utils.MetricLogger(delimiter="  ")
    for x in metric_logger.log_every(data_loader, print_freq=10, header=f'{prefix} Image metrics @{eval_size}:'):
        # Prepare clean and noised images
        clean_images = x[domain].clone().to(device)

        # Optionally mask out invalid regions and concat mask and images
        clean_images = mask_out_samples(clean_images, x.get('mask_valid', None), mask_value=mask_value)

        # Resize image to eval size
        clean_images = F.interpolate(clean_images, eval_size, mode='bilinear', align_corners=False)

        # Autoencode the images
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype, enabled=dtype != torch.float32):
            quant, _, tokens = unwrap_model(model).encode(clean_images)
            output = unwrap_model(model).decode_quant(
                quant, scheduler=noise_schedule, timesteps=num_diffusion_steps, 
                image_size=eval_size, verbose=False, orig_res=eval_res_cond
            )

        if log_codebook_usage:
            local_tokens.append(tokens.flatten().cpu())

        # Convert inputs and outputs to images again
        if domain in ['rgb', 'normal']:
            # 3-channel domains in [-1,1]
            gt = denormalize(clean_images[:,:3], mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
            reconst = denormalize(output[:,:3], mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        elif domain in ['depth']:
            # 1-channel per-sample standardized domains
            # TODO: Perform robust per-sample standardization to evaluate shift and scale invariant metrics
            gt = clean_images[:,:1]
            reconst = output[:,:1]
        elif domain in ['edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d', 'reshading']:
            # 1-channel domains in [-1,1]
            gt = denormalize(clean_images[:,:1], mean=(0.5,), std=(0.5,))
            reconst = denormalize(output[:,:1], mean=(0.5,), std=(0.5,))
        elif domain in ['principal_curvature']:
            # 2-channel domains in [-1,1]
            gt = denormalize(clean_images[:,:2], mean=(0.5,0.5), std=(0.5,0.5))
            reconst = denormalize(output[:,:2], mean=(0.5,0.5), std=(0.5,0.5))
        else:
            gt = clean_images.contiguous()
            reconst = output.contiguous()

        # Compute metrics
        mse_metric.update(reconst, gt)
        mae_metric.update(reconst, gt)
        psnr_metric.update(reconst, gt)
        ms_ssim_metric.update(reconst, gt)
        if fid_metric is not None:
            fid_metric.update(gt, real=True)
            fid_metric.update(reconst, real=False)
        if lpips_metric is not None:
            lpips_metric.update(reconst, gt)
        if inception_metric is not None:
            inception_metric.update(reconst)
            

    # Compute and log metrics
    results = {}
    prefix = f'{prefix} '
    suffix = f'@{eval_size}'

    results[prefix + 'MSE' + suffix] = mse_metric.compute().item()
    results[prefix + 'MAE' + suffix] = mae_metric.compute().item()
    results[prefix + 'PSNR' + suffix] = psnr_metric.compute().item()
    results[prefix + 'MS-SSIM' + suffix] = ms_ssim_metric.compute().item()
    if fid_metric is not None:
        results[prefix + 'FID' + suffix] = fid_metric.compute().item()
    if lpips_metric is not None:
        results[prefix + 'LPIPS' + suffix] = lpips_metric.compute().item()
    if inception_metric is not None:
        inception_mean, inception_std = inception_metric.compute()
        results[prefix + 'InceptionMean' + suffix] = inception_mean.item()
        results[prefix + 'InceptionStd' + suffix] = inception_std.item()

    # Reset metrics
    mse_metric.reset()
    mae_metric.reset()
    psnr_metric.reset()
    ms_ssim_metric.reset()
    if fid_metric is not None:
        fid_metric.reset()
    if lpips_metric is not None:
        lpips_metric.reset()
    if inception_metric is not None:
        inception_metric.reset()

    # Sync the tokens across processes to compute codebook usage
    if log_codebook_usage:
        local_tokens = torch.cat(local_tokens, dim=0)
        if utils.is_dist_avail_and_initialized():
            global_tokens = [torch.zeros_like(local_tokens) for _ in range(utils.get_world_size())]
            dist.all_gather_object(global_tokens, local_tokens)
            global_tokens = torch.cat(global_tokens, dim=0)
        else:
            global_tokens = local_tokens
        codebook_usage = compute_codebook_usage(
            global_tokens, 
            codebook_size=unwrap_model(model).quantize.codebook_size, 
            window_size=256 * tokens.shape[1:].numel() # Compute codebook usage over 256 images
        )
        results[prefix + 'CodebookUsage' + suffix] = codebook_usage

        # Log some additional codebook usage statistics to wandb
        if utils.is_main_process() and log_writer is not None:
            # Log the codebook EMA cluster sizes to wandb
            if hasattr(unwrap_model(model).quantize, '_codebook') and hasattr(unwrap_model(model).quantize._codebook, 'cluster_size'):
                fig = plt.figure(figsize=(8, 6))
                plt.hist(unwrap_model(model).quantize._codebook.cluster_size.cpu().numpy(), bins=100)
                plt.xlabel('EMA cluster size')
                plt.ylabel('Count')
                plt.title('Codebook EMA cluster sizes')
                log_writer.wandb_safe_log({prefix + 'CodebookEMAClusterSizes' + suffix: wandb.Image(fig)}, commit=False)
                plt.close(fig)

            # Log a plot of the number of times each token is used in the validation set to wandb
            fig = plt.figure(figsize=(8, 6))
            plt.plot(np.sort(np.bincount(global_tokens.flatten()))[::-1])
            plt.xscale('log')
            plt.xlabel('Token (sorted by frequency)')
            plt.ylabel('Count')
            plt.title('Token usage frequency')
            log_writer.wandb_safe_log({prefix + 'TokenUsageFreq' + suffix: wandb.Image(fig)}, commit=False)
            plt.close(fig)

        # Reset the codebook utilization statistics
        del local_tokens, global_tokens

    metric_logger.update(**results)
    # Gather the stats from all processes (they should already be the same since we sync the torcheval metrics after every step)
    metric_logger.synchronize_between_processes()
    print(f"{prefix} Generation results:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def make_grid(images: List[Image.Image]) -> Image.Image:
    """Aggregate a list of PIL images into a grid of images."""
    cols = int(np.floor(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / cols))
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid     


@torch.no_grad()
def eval_image_log(model: Union[nn.Module, DDP], 
                   data_loader: Iterable, 
                   device: Union[torch.device, str],
                   domain: str, 
                   eval_size: int, 
                   noise_schedule: SchedulerMixin,
                   num_diffusion_steps: int,
                   dtype: torch.dtype = torch.float16, 
                   num_logged_images: int = 100, 
                   mask_value: Optional[float] = None, 
                   eval_res_cond: Optional[int] = None,
                   log_writer: Optional[utils.WandbLogger] = None,
                   prefix: str = '[Eval]') -> None:
    """Log several reconstructed images to wandb.

    Args:
        model: Model to evaluate.
        data_loader: Validation set data loader.
        device: Device to evaluate on.
        domain: Image domain.
        eval_size: Resolution to evaluate at.
        noise_schedule: Noise schedule to use for diffusion.
        num_diffusion_steps: Number of diffusion steps to use for eval.
        dtype: Data type for mixed precision inference.
        num_logged_images: Number of images to log.
        mask_value: Value to mask out invalid regions with.
        eval_res_cond: A fixed eval image size to condition the diffusion on. 
          Ignored if None. See SDXL https://arxiv.org/abs/2307.01952 for more details.
        log_writer: Wandb logger.
        prefix: Prefix for wandb logging.
    """

    if log_writer is None:
        print('No wandb logger provided, skipping image logging.')
        return
    
    model.eval()
    
    gt_imgs = []
    reconst_imgs = []


    if utils.is_main_process():

        for x in data_loader:
            # Prepare clean and noised images
            clean_images = x[domain].clone().to(device)

            # Optionally mask out invalid regions and concat mask and images
            clean_images = mask_out_samples(clean_images, x.get('mask_valid', None), mask_value=mask_value)

            # Resize image to eval size
            clean_images = F.interpolate(clean_images, eval_size, mode='bilinear', align_corners=False)

            # Autoencode the images
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype, enabled=dtype != torch.float32):
                output = unwrap_model(model).autoencode(
                    clean_images, scheduler=noise_schedule, timesteps=num_diffusion_steps, verbose=False, orig_res=eval_res_cond
                )

            # Convert inputs and outputs to images again
            if domain in ['rgb', 'normal']:
                gt = denormalize(clean_images[:,:3], mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
                reconst = denormalize(output[:,:3], mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
            elif domain in ['depth']:
                # 1-channel per-sample standardized domains
                gt = clean_images[:,:1]
                batch_min = rearrange(gt, 'b c h w -> b (c h w)').min(1)[0][:,None,None,None]
                batch_max = rearrange(gt, 'b c h w -> b (c h w)').max(1)[0][:,None,None,None]
                gt = (gt - batch_min) / (batch_max - batch_min)
                reconst = (output[:,:1] - batch_min) / (batch_max - batch_min)
            elif domain in ['edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d', 'reshading']:
                # 1-channel domains in [-1,1]
                gt = denormalize(clean_images[:,:1], mean=(0.5,), std=(0.5,))
                reconst = denormalize(output[:,:1], mean=(0.5,), std=(0.5,))
            elif domain in ['principal_curvature']:
                # 2-channel domains in [-1,1]
                B, _, H, W = clean_images.shape
                gt = torch.zeros(B, 3, H, W, device=clean_images.device, dtype=clean_images.dtype)
                gt[:,[0,1]] = denormalize(clean_images[:,:2], mean=(0.5,0.5), std=(0.5,0.5))
                reconst = torch.zeros(B, 3, H, W, device=clean_images.device, dtype=clean_images.dtype)
                reconst[:,[0,1]] = denormalize(output[:,:2], mean=(0.5,0.5), std=(0.5,0.5))
            else:
                raise NotImplementedError

            # Save some example images
            if domain in ['rgb', 'normal']:
                gt_bytes = (255 * gt.float().permute(0,2,3,1).clamp(0,1).cpu().numpy()).astype(np.uint8)
                reconst_bytes = (255 * reconst.float().permute(0,2,3,1).clamp(0,1).cpu().numpy()).astype(np.uint8)
       
            gt_imgs.extend([Image.fromarray(b) for b in gt_bytes])
            reconst_imgs.extend([Image.fromarray(b) for b in reconst_bytes])

        # Log example images to wandb
        if len(gt_imgs) > 0:
            log_writer.wandb_safe_log({f'{prefix} GT @{eval_size}': wandb.Image(make_grid(gt_imgs[:num_logged_images]))}, commit=False)
            log_writer.wandb_safe_log({f'{prefix} Reconst @{eval_size}': wandb.Image(make_grid(reconst_imgs[:num_logged_images]))}, commit=False)
        
        print(f"Logged {num_logged_images} eval images @ res {eval_size}")


if __name__ == '__main__':
    args = get_args()
    
    utils.setup_run_name(args)
    utils.setup_s3_args(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
