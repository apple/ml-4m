from typing import Optional, List
import os
import math
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import requests
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
from torchvision.transforms.functional import center_crop

from fourm.models.fm import FM
from fourm.vq.vqvae import VQVAE, DiVAE
from fourm.models.generate import GenerationSampler, build_chained_generation_schedules, init_empty_target_modality, init_full_input_modality, custom_text
from fourm.utils.plotting_utils import decode_dict
from fourm.data.modality_info import MODALITY_INFO
from fourm.data.modality_transforms import RGBTransform
from fourm.utils import load_safetensors
from fourm.utils.plotting_utils import decode_dict, visualize_bboxes, plot_text_in_square, text_to_pil_image

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


# Default chained generation order
DEFAULT_ORDER = [
    'tok_clip@224', 'tok_dinov2@224', 'tok_imagebind@224', 'tok_depth@224', 'tok_normal@224', 
    'tok_semseg@224', 'tok_canny_edge@224', 'tok_sam_edge@224', 'tok_rgb@224', 
    'caption', 'det', 'human_poses', 'sam_instance', 'color_palette', 'metadata',
]

# Default super-resolution chained generation order
DEFAULT_ORDER_SR = [
    'tok_clip@448', 'tok_depth@448', 'tok_normal@448', 
    'tok_semseg@448', 'tok_rgb@448',
]

# Default generation parameters for the case where the input contains RGB
DEFAULTS_RGB2X = {
    'tok_clip@224/tok_depth@224/tok_normal@224/tok_semseg@224/tok_canny_edge@224/tok_sam_edge@224': {
        'tokens_per_target': 196, 'autoregression_scheme': 'roar', 'decoding_steps': 1,
        'token_decoding_schedule': 'linear', 'temp': 0.01, 'temp_schedule': 'constant',
        'cfg_scale': 2.0, 'cfg_schedule': 'constant',
    },
    'tok_dinov2@224/tok_imagebind@224': {
        'tokens_per_target': 256, 'autoregression_scheme': 'roar', 'decoding_steps': 1,
        'token_decoding_schedule': 'linear', 'temp': 0.01, 'temp_schedule': 'constant',
        'cfg_scale': 2.0, 'cfg_schedule': 'constant',
    },
    'caption/det': {
        'tokens_per_target': 256, 'autoregression_scheme': 'autoregressive', 'decoding_steps': None,
        'token_decoding_schedule': None, 'temp': 0.3, 'temp_schedule': 'constant',
        'cfg_scale': 1.0, 'cfg_schedule': 'constant',
    },
    'human_poses': {
        'tokens_per_target': 275, 'autoregression_scheme': 'autoregressive', 'decoding_steps': None,
        'token_decoding_schedule': None, 'temp': 0.1, 'temp_schedule': 'constant',
        'cfg_scale': 1.0, 'cfg_schedule': 'constant',
    },
    'sam_instance': {
        'tokens_per_target': 256, 'autoregression_scheme': 'autoregressive', 'decoding_steps': None,
        'token_decoding_schedule': None, 'temp': 0.01, 'temp_schedule': 'constant',
        'cfg_scale': 1.0, 'cfg_schedule': 'constant',
    },
    'color_palette': {
        'tokens_per_target': 23, 'autoregression_scheme': 'autoregressive', 'decoding_steps': None,
        'token_decoding_schedule': None, 'temp': 0.1, 'temp_schedule': 'constant',
        'cfg_scale': 1.0, 'cfg_schedule': 'constant',
    },
    'metadata': {
        'tokens_per_target': 40, 'autoregression_scheme': 'autoregressive', 'decoding_steps': None,
        'token_decoding_schedule': None, 'temp': 0.1, 'temp_schedule': 'constant',
        'cfg_scale': 1.0, 'cfg_schedule': 'constant',
    },
}

# Default generation parameters for the case where the target is RGB
DEFAULTS_X2RGB = {
    'tok_clip@224': {
        'tokens_per_target': 196, 'autoregression_scheme': 'roar', 'decoding_steps': 50,
        'token_decoding_schedule': 'linear', 'temp': 5.0, 'temp_schedule': 'onex:0.5:0.5',
        'cfg_scale': 3.0, 'cfg_schedule': 'constant',
    },
    'tok_dinov2@224/tok_imagebind@224': {
        'tokens_per_target': 256, 'autoregression_scheme': 'roar', 'decoding_steps': 8,
        'token_decoding_schedule': 'linear', 'temp': 0.01, 'temp_schedule': 'constant',
        'cfg_scale': 2.0, 'cfg_schedule': 'constant',
    },
    'tok_depth@224/tok_normal@224/tok_semseg@224/tok_canny_edge@224/tok_sam_edge@224': {
        'tokens_per_target': 196, 'autoregression_scheme': 'roar', 'decoding_steps': 8,
        'token_decoding_schedule': 'linear', 'temp': 3.0, 'temp_schedule': 'onex:0.5:0.5',
        'cfg_scale': 2.0, 'cfg_schedule': 'constant',
    },
    'tok_rgb@224': {
        'tokens_per_target': 196, 'autoregression_scheme': 'roar', 'decoding_steps': 25,
        'token_decoding_schedule': 'linear', 'temp': 3.0, 'temp_schedule': 'onex:0.5:0.5',
        'cfg_scale': 2.0, 'cfg_schedule': 'constant',
    },
    'caption/det': {
        'tokens_per_target': 256, 'autoregression_scheme': 'autoregressive', 'decoding_steps': None,
        'token_decoding_schedule': None, 'temp': 0.3, 'temp_schedule': 'constant',
        'cfg_scale': 1.0, 'cfg_schedule': 'constant',
    },
    'human_poses': {
        'tokens_per_target': 275, 'autoregression_scheme': 'autoregressive', 'decoding_steps': None,
        'token_decoding_schedule': None, 'temp': 0.1, 'temp_schedule': 'constant',
        'cfg_scale': 1.0, 'cfg_schedule': 'constant',
    },
    'sam_instance': {
        'tokens_per_target': 256, 'autoregression_scheme': 'autoregressive', 'decoding_steps': None,
        'token_decoding_schedule': None, 'temp': 0.01, 'temp_schedule': 'constant',
        'cfg_scale': 1.0, 'cfg_schedule': 'constant',
    },
    'color_palette': {
        'tokens_per_target': 23, 'autoregression_scheme': 'autoregressive', 'decoding_steps': None,
        'token_decoding_schedule': None, 'temp': 0.1, 'temp_schedule': 'constant',
        'cfg_scale': 1.0, 'cfg_schedule': 'constant',
    },
    'metadata': {
        'tokens_per_target': 40, 'autoregression_scheme': 'autoregressive', 'decoding_steps': None,
        'token_decoding_schedule': None, 'temp': 0.1, 'temp_schedule': 'constant',
        'cfg_scale': 1.0, 'cfg_schedule': 'constant',
    },
}

# Default generation parameters for super-resolution
DEFAULTS_SR = {
    'tok_clip@448/tok_depth@448/tok_normal@448/tok_semseg@448/tok_rgb@448': {
        'tokens_per_target': 784, 'autoregression_scheme': 'maskgit', 'decoding_steps': 8,
        'token_decoding_schedule': 'cosine', 'temp': 1.0, 'temp_schedule': 'constant',
        'cfg_scale': 2.0, 'cfg_schedule': 'constant',
    },
}

# Plotting names for each modality
MODALITY_PLOTTING_NAME_MAP = {
    'caption': 'Caption', 
    'det': 'Bounding boxes', 
    'human_poses': 'Human poses', 
    'sam_instance': 'SAM instances (single pass)', 
    'color_palette': 'Color palette', 
    'metadata': 'Metadata',
    'rgb@224': 'RGB (224x224)',
    'rgb@448': 'RGB (448x448)',
    'tok_rgb@224': 'RGB (tokenized, 224x224)',  
    'tok_rgb@448': 'RGB (tokenized, 448x448)', 
    'tok_clip@224': 'CLIP-B/16 (224x224)', 
    'tok_clip@448': 'CLIP-B/16  (448x448)', 
    'tok_depth@224': 'Depth (224x224)', 
    'tok_depth@448': 'Depth (448x448)', 
    'tok_normal@224': 'Normals (224x224)', 
    'tok_normal@448': 'Normals (448x448)', 
    'tok_semseg@224': 'Semantic segmentation (224x224)', 
    'tok_semseg@448': 'Semantic segmentation (448x448)', 
    'tok_canny_edge@224': 'Canny edges (224x224)', 
    'tok_sam_edge@224': 'SAM edges (224x224)', 
    'tok_dinov2@224': 'DINOv2-B/14 (224x224)', 
    'tok_imagebind@224': 'ImageBind-H/14 (224x224)', 
}

# Optional fixed plotting order (by default, plotting order is determined by generation order)
MODALITY_PLOTTING_ORDER = [
    'rgb@224', 'rgb@448', 'tok_rgb@224', 'tok_rgb@448',
    'tok_depth@224', 'tok_depth@448', 'tok_normal@224', 'tok_normal@448', 
    'tok_semseg@224', 'tok_semseg@448', 'tok_canny_edge@224', 'tok_sam_edge@224', 
    'sam_instance', 'human_poses', 'det', 'caption', 'metadata', 'color_palette',
    'tok_clip@224', 'tok_clip@448', 'tok_dinov2@224', 'tok_imagebind@224',
]


def get_value(defaults_dict, domain, key):
    """Look up a default value belonging to a given domain and key."""
    for domains, defaults in defaults_dict.items():
        if domain in domains:
            return defaults[key]

def load_model(model_id, model_class):
    """Load a model from HuggingFace hub or a given .safetensors checkpoint path."""
    if model_id.endswith('.safetensors'):
        ckpt, config = load_safetensors(model_id)
        model = model_class(config=config)
        model.load_state_dict(ckpt)
    else:
        model = model_class.from_pretrained(model_id)
    return model

def img_from_url(url: str):
    rgb_transform = RGBTransform(imagenet_default_mean_and_std=True)
    img_data = requests.get(url).content
    with open('demo.png', 'wb') as handler:
        handler.write(img_data)
    img_pil = rgb_transform.load('./demo.png')
    img_pil = rgb_transform.preprocess(img_pil)
    img_pil = center_crop(img_pil, (min(img_pil.size), min(img_pil.size))).resize((224,224))
    img = rgb_transform.postprocess(img_pil).unsqueeze(0)
    return img


class Demo4MSampler(nn.Module):
    """Convenience wrapper for easy 4M loading and generation. Users can specify HuggingFace Hub 
    model URLs, or downloaded safetensors checkpoints paths, and the models will be automatically 
    loaded. The `forward` function can be used for RGB-2-all and {caption,det}-2-all generation. 
    This wrapper is only intended for quickly trying out 4M models. For more advanced usecases we 
    recommend looking at the generation notebooks in `./notebooks/`, and `./run_generation.py`.
    
    Args:
        fm: Hub or safetensors path of 4M base model
        fm_sr: Hub or safetensors path of 4M super-resolution model
        tok_rgb: Hub or safetensors path of RGB tokenizer
        tok_depth: Hub or safetensors path of depth tokenizer
        tok_normal: Hub or safetensors path of surface normal tokenizer
        tok_edge: Hub or safetensors path of canny edge tokenizer (for SAM and RGB edges)
        tok_semseg: Hub or safetensors path of COCO semantic segmentation tokenizer
        tok_clip: Hub or safetensors path of CLIP-B/16 tokenizer
        tok_dinov2: Hub or safetensors path of DINOv2-B/14 tokenizer
        tok_imagebind: Hub or safetensors path of ImageBind-H/14 tokenizer
        tok_sam_instance: Hub or safetensors path of SAM instance tokenizer
        tok_human_poses: Hub or safetensors path of human poses tokenizer
        tok_text: Path to text tokenizer JSON file
        mods: Optional list of modalities to override default behavior of generating everything
        mods_sr: Optional list of super-res modalities to override default behavior of generating everything
    """
    def __init__(self, 
                 fm: str = 'EPFL-VILAB/4M-21_XL_CC12M', 
                 fm_sr: Optional[str] = 'EPFL-VILAB/4M-7-SR_L_CC12M', 
                 tok_rgb: Optional[str] = 'EPFL-VILAB/4M_tokenizers_rgb_16k_224-448',
                 tok_depth: Optional[str] = 'EPFL-VILAB/4M_tokenizers_depth_8k_224-448',
                 tok_normal: Optional[str] = 'EPFL-VILAB/4M_tokenizers_normal_8k_224-448',
                 tok_edge: Optional[str] = 'EPFL-VILAB/4M_tokenizers_edge_8k_224-512',
                 tok_semseg: Optional[str] = 'EPFL-VILAB/4M_tokenizers_semseg_4k_224-448',
                 tok_clip: Optional[str] = 'EPFL-VILAB/4M_tokenizers_CLIP-B16_8k_224-448',
                 tok_dinov2: Optional[str] = 'EPFL-VILAB/4M_tokenizers_DINOv2-B14_8k_224-448',
                 tok_imagebind: Optional[str] = 'EPFL-VILAB/4M_tokenizers_ImageBind-H14_8k_224-448',
                 tok_sam_instance: Optional[str] = 'EPFL-VILAB/4M_tokenizers_sam-instance_1k_64',
                 tok_human_poses: Optional[str] = 'EPFL-VILAB/4M_tokenizers_human-poses_1k_8',
                 tok_text: str = './fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json',
                 mods: Optional[List[str]] = None,
                 mods_sr: Optional[List[str]] = None,
                 verbose: bool = True):
        super().__init__()

        self.verbose = verbose
        if self.verbose:
            print('Loading 4M models and tokenizers...', end='')

        # Load 4M model and initialize sampler
        fm = load_model(fm, FM)
        self.sampler_fm = GenerationSampler(fm)
        self.mods = mods or list(set(fm.encoder_modalities) | set(fm.decoder_modalities))

        # Load optional 4M super-res model and initialize sampler
        if fm_sr is not None:
            fm_sr = load_model(fm_sr, FM)
            self.sampler_fm_sr = GenerationSampler(fm_sr)
            self.mods_sr = mods_sr or list(set(fm_sr.encoder_modalities) | set(fm_sr.decoder_modalities))
        else:
            self.sampler_fm_sr = None

        # Load tokenizers
        self.toks = {}
        if ('tok_rgb@224' in self.mods or 'tok_rgb@448' in self.mods_sr) and tok_rgb is not None:
            self.toks['tok_rgb'] = load_model(tok_rgb, DiVAE)
        if ('tok_depth@224' in self.mods or 'tok_depth@448' in self.mods_sr) and tok_depth is not None:
            self.toks['tok_depth'] = load_model(tok_depth, DiVAE)
        if ('tok_normal@224' in self.mods or 'tok_normal@448' in self.mods_sr) and tok_normal is not None:
            self.toks['tok_normal'] = load_model(tok_normal, DiVAE)
        if ('tok_canny_edge@224' in self.mods or 'tok_sam_edge@224' in self.mods) and tok_edge is not None:
            self.toks['tok_canny_edge'] = load_model(tok_edge, DiVAE)
            self.toks['tok_sam_edge'] = self.toks['tok_canny_edge'] # Shared tokenizer
        if ('tok_semseg@224' in self.mods or 'tok_semseg@448' in self.mods_sr) and tok_semseg is not None:
            self.toks['tok_semseg'] = load_model(tok_semseg, VQVAE)
        if ('tok_clip@224' in self.mods or 'tok_clip@448' in self.mods_sr) and tok_clip is not None:
            self.toks['tok_clip'] = load_model(tok_clip, VQVAE)
        if 'tok_dinov2@224' in self.mods and tok_dinov2 is not None:
            self.toks['tok_dinov2'] = load_model(tok_dinov2, VQVAE)
        if 'tok_imagebind@224' in self.mods and tok_imagebind is not None:
            self.toks['tok_imagebind'] = load_model(tok_imagebind, VQVAE)
        if 'sam_instance' in self.mods and tok_sam_instance is not None:
            self.toks['sam_instance'] = load_model(tok_sam_instance, VQVAE)
        if 'human_poses' in self.mods and tok_human_poses is not None:
            self.toks['human_poses'] = load_model(tok_human_poses, VQVAE)
        self.toks = nn.ModuleDict(self.toks)
        self.tok_text = Tokenizer.from_file(tok_text)

        if self.verbose:
            print(' done!')

    @property
    def device(self):
        return next(self.parameters()).device

    def __setup_conds_and_targets(self, sample):
        # Input and output modalities
        cond_domains = [domain for domain in list(sample.keys()) if domain in self.mods]
        target_domains = [domain for domain in DEFAULT_ORDER if (domain not in cond_domains and domain in self.mods)]
        if 'rgb@224' in cond_domains:
            # Do not generate tokenized RGB if pixel RGB is given as input
            target_domains.remove('tok_rgb@224')
        return cond_domains, target_domains

    def __setup_sr_conds_and_targets(self, sample):
        cond_domains_sr = [domain for domain in list(sample.keys()) if domain in self.mods_sr]
        target_domains_sr = [domain for domain in DEFAULT_ORDER_SR if (domain.replace('448', '224') in cond_domains_sr and domain in self.mods_sr)]
        return cond_domains_sr, target_domains_sr

    def __setup_sample_and_schedule(self, sample, cond_domains, target_domains, cfg_grow_conditioning=True):
        # 1 - Setup generation schedule
        
        defaults = DEFAULTS_RGB2X if ('rgb@224' in cond_domains or 'tok_rgb@224' in cond_domains) else DEFAULTS_X2RGB

        tokens_per_target = [get_value(defaults, domain, 'tokens_per_target') for domain in target_domains]
        autoregression_schemes = [get_value(defaults, domain, 'autoregression_scheme') for domain in target_domains]
        decoding_steps = [get_value(defaults, domain, 'decoding_steps') for domain in target_domains]
        token_decoding_schedules = [get_value(defaults, domain, 'token_decoding_schedule') for domain in target_domains]
        temps = [get_value(defaults, domain, 'temp') for domain in target_domains]
        temp_schedules = [get_value(defaults, domain, 'temp_schedule') for domain in target_domains]
        cfg_scales = [get_value(defaults, domain, 'cfg_scale') for domain in target_domains]
        cfg_schedules = [get_value(defaults, domain, 'cfg_schedule') for domain in target_domains]
        
        schedule = build_chained_generation_schedules(
            cond_domains=cond_domains, target_domains=target_domains, tokens_per_target=tokens_per_target, 
            autoregression_schemes=autoregression_schemes, decoding_steps=decoding_steps, 
            token_decoding_schedules=token_decoding_schedules, temps=temps, temp_schedules=temp_schedules,
            cfg_scales=cfg_scales, cfg_schedules=cfg_schedules, cfg_grow_conditioning=cfg_grow_conditioning, 
        )

        # 2 - Setup sample
        
        sample_dict = {}

        # Handle special cases
        if 'caption' in sample:
            caption = sample.pop('caption')
            sample_dict = custom_text(
                sample_dict, input_text=caption, eos_token='[EOS]', 
                key='caption', device=self.device, text_tokenizer=self.tok_text
            )
        if 'det' in sample:
            caption = sample.pop('det')
            sample_dict = custom_text(
                sample_dict, input_text=caption, eos_token='[EOS]', 
                key='det', device=self.device, text_tokenizer=self.tok_text
            )
        # Add remaining modalities
        sample_dict.update({domain: {'tensor': tensor} for domain, tensor in sample.items()})
        
        # Initialize these remaining input modalities (caption and det are already initialized by custom_text)
        for cond_mod in sample.keys():
            sample_dict = init_full_input_modality(sample_dict, MODALITY_INFO, cond_mod, self.device, eos_id=self.tok_text.token_to_id("[EOS]"))
        
        # Initialize target modalities
        for target_mod, ntoks in zip(target_domains, tokens_per_target):
            sample_dict = init_empty_target_modality(sample_dict, MODALITY_INFO, target_mod, 1, ntoks, self.device)

        return sample_dict, schedule

    def __setup_sr_sample_and_schedule(self, out_dict, cond_domains_sr, target_domains_sr, cfg_grow_conditioning_sr=True):
        # 1 - Setup generation schedule
        
        tokens_per_target_sr = [get_value(DEFAULTS_SR, domain, 'tokens_per_target') for domain in target_domains_sr]
        autoregression_schemes_sr = [get_value(DEFAULTS_SR, domain, 'autoregression_scheme') for domain in target_domains_sr]
        decoding_steps_sr = [get_value(DEFAULTS_SR, domain, 'decoding_steps') for domain in target_domains_sr]
        token_decoding_schedules_sr = [get_value(DEFAULTS_SR, domain, 'token_decoding_schedule') for domain in target_domains_sr]
        temps_sr = [get_value(DEFAULTS_SR, domain, 'temp') for domain in target_domains_sr]
        temp_schedules_sr = [get_value(DEFAULTS_SR, domain, 'temp_schedule') for domain in target_domains_sr]
        cfg_scales_sr = [get_value(DEFAULTS_SR, domain, 'cfg_scale') for domain in target_domains_sr]
        cfg_schedules_sr = [get_value(DEFAULTS_SR, domain, 'cfg_schedule') for domain in target_domains_sr]
        
        schedule_sr = build_chained_generation_schedules(
            cond_domains=cond_domains_sr, target_domains=target_domains_sr, tokens_per_target=tokens_per_target_sr, 
            autoregression_schemes=autoregression_schemes_sr, decoding_steps=decoding_steps_sr, 
            token_decoding_schedules=token_decoding_schedules_sr, temps=temps_sr, temp_schedules=temp_schedules_sr,
            cfg_scales=cfg_scales_sr, cfg_schedules=cfg_schedules_sr, cfg_grow_conditioning=cfg_grow_conditioning_sr, 
        )

        # 2 - Setup sample

        sample_sr = out_dict

        # Handle case where generated caption or bounding boxes is just [EOS]
        if 'caption' in sample_sr and sample_sr['caption']['tensor'].shape[1] <= 1 and 'caption' in cond_domains_sr:
            sample_sr = custom_text(
                sample_sr, input_text='[S_1]', eos_token='[EOS]', 
                key='caption', device=self.device, text_tokenizer=self.tok_text
            )
        if 'det' in sample_sr and sample_sr['det']['tensor'].shape[1] <= 1 and 'det' in cond_domains_sr:
            sample_sr = custom_text(
                sample_sr, input_text='[S_1]', eos_token='[EOS]', 
                key='det', device=self.device, text_tokenizer=self.tok_text
            )

        # Initialize input modalities
        for cond_mod in cond_domains_sr:
            sample_sr = init_full_input_modality(sample_sr, MODALITY_INFO, cond_mod, self.device, eos_id=self.tok_text.token_to_id("[EOS]"))
        
        # Initialize target modalities
        for target_mod, ntoks in zip(target_domains_sr, tokens_per_target_sr):
            sample_sr = init_empty_target_modality(sample_sr, MODALITY_INFO, target_mod, 1, ntoks, self.device)
        
        return sample_sr, schedule_sr

    def forward(self, sample, seed: Optional[int] = None, top_p: float = 0.8, top_k: float = 0.0, target_modalities: Optional[List[str]] = None, perform_sr: bool = True):
        seed = seed or np.random.randint(np.iinfo(np.int64).max)
        
        # Prepare the generation parameters and sample
        cond_domains, target_domains = self.__setup_conds_and_targets(sample)
        target_domains = target_modalities or target_domains
        sample, generation_schedule = self.__setup_sample_and_schedule(sample, cond_domains, target_domains)

        # Generation and decoding at the base resolution 224x224
        if self.verbose:
            print(f'Generating {cond_domains} -> {target_domains} ...')
        out_dict = self.sampler_fm.generate(
            sample, generation_schedule, text_tokenizer=self.tok_text, 
            verbose=self.verbose, seed=seed, top_p=top_p, top_k=top_k,
        )
        dec_dict = decode_dict(
            out_dict, self.toks, self.tok_text, image_size=224, 
            patch_size=16, decoding_steps=50
        )
        
        # Optional upsampling to 448x448
        if self.sampler_fm_sr is not None and perform_sr:
            cond_domains_sr, target_domains_sr = self.__setup_sr_conds_and_targets(out_dict)
            sample_sr, generation_schedule_sr = self.__setup_sr_sample_and_schedule(out_dict, cond_domains_sr, target_domains_sr)
    
            if self.verbose:
                print(f'Super-resolving {target_domains_sr} ...')
            out_dict_sr = self.sampler_fm_sr.generate(
                sample_sr, generation_schedule_sr, text_tokenizer=self.tok_text, 
                verbose=self.verbose, seed=seed+1, top_p=top_p, top_k=top_k,
            )
            dec_dict = decode_dict(
                out_dict_sr, self.toks, self.tok_text, image_size=448, 
                patch_size=16, decoding_steps=50
            )

        # Remove padding tokens
        if 'caption' in dec_dict:
            dec_dict['caption'][0].replace('[PAD]', '').strip()
        if 'det' in dec_dict:
            dec_dict['det'][0].replace('[PAD]', '').strip()
        
        return dec_dict
    
    def plot_modalities(self, mod_dict, ncols_max=5, figscale=4.0, save_path=None, use_fixed_plotting_order=False):
        nmods = len(mod_dict)
        ncols = min(nmods, ncols_max)
        nrows = math.ceil(nmods / ncols)
        
        fig, ax = plt.subplots(
            nrows=nrows, ncols=ncols, 
            figsize=(ncols*figscale, nrows*figscale), 
            facecolor=(1, 1, 1)
        )

        if use_fixed_plotting_order:
            mod_dict = {
                k: mod_dict[k] for k in MODALITY_PLOTTING_ORDER
                if k in mod_dict
            }

        for i, (mod_name, mod) in enumerate(mod_dict.items()):
            if nrows == 1:
                ax_i = ax[i]
            else:
                row, col = i // ncols, i % ncols
                ax_i = ax[row,col]

            if mod_name == 'det':
                # Attempt to get the first available value from mod_dict according to the priority
                keys_in_order = ['rgb@448', 'rgb@224', 'tok_rgb@448', 'tok_rgb@224']
                rgb_background = next((mod_dict[key] for key in keys_in_order if key in mod_dict), np.ones((224, 224, 3)))
                rgb_background = (255 * rgb_background).astype(np.uint8)
                ax_i.imshow(visualize_bboxes(rgb_background, mod[0],).astype(np.uint8))
            elif mod_name == 'caption':
                plot_text_in_square(ax_i, mod[0], wrap_width=16, fontsize=14)
            elif mod_name == 'metadata':
                metadata_pred = ',\n'.join([f'{k}: {v:.2f}' if isinstance(v, float) else f'{k}: {v}' for k, v in mod.items()])
                plot_text_in_square(ax_i, metadata_pred, wrap_width=36, fontsize=13)
            else:
                ax_i.imshow(mod)
            
            ax_i.set_title(MODALITY_PLOTTING_NAME_MAP.get(mod_name, mod_name), fontsize=18)

        for i, axis in enumerate(ax.flatten()):
            axis.set_xticks([])
            axis.set_yticks([])
            if i >= len(mod_dict):
                axis.spines['top'].set_visible(False)
                axis.spines['right'].set_visible(False)
                axis.spines['bottom'].set_visible(False)
                axis.spines['left'].set_visible(False)
        
        plt.tight_layout()
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def modalities_to_pil(self, mod_dict, use_fixed_plotting_order=False, resize=None):
        if use_fixed_plotting_order:
            mod_dict = {
                k: mod_dict[k] for k in MODALITY_PLOTTING_ORDER
                if k in mod_dict
            }

        plotted_modalities = []

        for i, (mod_name, mod) in enumerate(mod_dict.items()):
            if mod_name == 'det':
                # Attempt to get the first available value from mod_dict according to the priority
                keys_in_order = ['rgb@448', 'rgb@224', 'tok_rgb@448', 'tok_rgb@224']
                rgb_background = next((mod_dict[key] for key in keys_in_order if key in mod_dict), np.ones((224, 224, 3)))
                rgb_background = (255 * rgb_background).astype(np.uint8)
                img_pil = Image.fromarray(visualize_bboxes(rgb_background, mod[0],).astype(np.uint8))
            elif mod_name == 'caption':
                img_pil = text_to_pil_image(mod[0][:512], wrap_width=40, fontsize=14)
            elif mod_name == 'metadata':
                metadata_pred = ',\n'.join([f'{k}: {v:.2f}' if isinstance(v, float) else f'{k}: {v}' for k, v in mod.items()])
                img_pil = text_to_pil_image(metadata_pred, wrap_width=36, fontsize=13)
            else:
                img_pil = Image.fromarray((255*mod).astype(np.uint8))

            if resize is not None:
                if mod_name in ['tok_clip@224', 'tok_dinov2@224', 'tok_imagebind@224', 'tok_clip@448']:
                    resample_mode = Image.Resampling.NEAREST
                else:
                    resample_mode = Image.Resampling.BILINEAR
                img_pil = img_pil.resize((resize, resize), resample=resample_mode)
            
            plot_name = MODALITY_PLOTTING_NAME_MAP.get(mod_name, mod_name)
            plotted_modalities.append((img_pil, plot_name))

        return plotted_modalities