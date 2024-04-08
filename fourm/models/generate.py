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
from collections import defaultdict
from typing import Union, List, Optional

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn
import torch.nn.functional as F

from fourm.utils import get_sentinel_to_id_mapping, merge_span_masking
from fourm.utils.generation import cosine_schedule, linear_schedule, onex_temp_schedule, linear_temp_schedule, continue_schedule
from tqdm import tqdm
import copy



def empty_img_modality(mod_dict, key):
    # Input mask
    mod_dict[key]['input_mask'][:] = True
    
    # Target Mask
    mod_dict[key]['target_mask'][:] = False
    
    return mod_dict

def empty_seq_modality(mod_dict, key, s1_id=5):
    # To create an empty sequence, we suppose an input budget of 1, and the rest assigned to targets

    # Input tensor
    # Input is [S_1], target is [S_1] ...... [S_2]
    # (so [S_1] [S_1] ..... [S_2] when combined)
    mod_dict[key]['tensor'][:] = 0
    mod_dict[key]['tensor'][:,[0,1]] = s1_id # s1_id is id of the first sentinel token ([S_1])
    mod_dict[key]['tensor'][:,-1] = s1_id + 1

    # Input mask
    # Set first token to input (i.e. 0), rest to target (i.e. 1)
    mod_dict[key]['input_mask'][:] = True
    mod_dict[key]['input_mask'][:,0] = False
    
    # Target Mask
    mod_dict[key]['target_mask'] = ~mod_dict[key]['input_mask']

    # Decoder attn mask
    # WARNING: Not needed / used in GenerationSampler, where causal mask is enforced
    # First token is input, not part of target
    mod_dict[key]['decoder_attention_mask'][:] = 1
    mod_dict[key]['decoder_attention_mask'][:, 0] = 0

    return mod_dict

def empty_seq_emb_modality(mod_dict, key):
     # Tensor
    mod_dict[key]['tensor'] = torch.zeros_like(mod_dict[key]['tensor'])
    
    # Input mask
    mod_dict[key]['input_mask'] = torch.ones_like(mod_dict[key]['input_mask'])
    # It is crucial to specify the input mask as such, CFG won't work otherwise!
    mod_dict[key]['input_mask'][:, 0] = False
    
    # Target Mask
    mod_dict[key]['target_mask'] = torch.ones_like(mod_dict[key]['target_mask'])
    
    # Decoder attn mask
    mod_dict[key]['decoder_attention_mask'][:] = False
    
    return mod_dict
    

def init_empty_target_modality(mod_dict, modality_info, domain, batch_size, num_tokens, device):
    """
    Initializes an empty target modality dictionary for a given domain. 
    Used to initialize target modality dictionaries for generation. 
    """
    if modality_info[domain]['type'] == 'img':
        # Initialize mod dict
        mod_dict[domain] = {
            'tensor': torch.zeros((batch_size, num_tokens), dtype=torch.int64, device=device),
            'input_mask': torch.ones((batch_size, num_tokens), dtype=torch.bool, device=device),
            'target_mask': torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=device),
        }
        # Set it to the correct values
        mod_dict = empty_img_modality(mod_dict, domain)

    elif modality_info[domain]['type'] in ['seq', 'seq_token', 'seq_emb']:
        # Initialize mod dict
        mod_dict[domain] = {
            'tensor': torch.zeros((batch_size, num_tokens), dtype=torch.int32, device=device),
            'input_mask': torch.ones((batch_size, num_tokens), dtype=torch.bool, device=device),
            'target_mask': torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=device),
            'decoder_attention_mask': torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=device),
        }
        # Set it to the correct values
        if modality_info[domain]['type'] in ['seq', 'seq_token']:
            mod_dict = empty_seq_modality(mod_dict, domain)
        elif modality_info[domain]['type'] == 'seq_emb':
            mod_dict = empty_seq_emb_modality(mod_dict, domain)
    else:
        raise ValueError()
        
    return mod_dict

def init_full_input_modality(mod_dict, modality_info, domain, device, eos_id=3):
    if 'input_mask' not in mod_dict[domain]:
        mod_dict[domain]['input_mask'] = torch.zeros(mod_dict[domain]['tensor'].shape, dtype=torch.bool, device=device)
    if 'target_mask' not in mod_dict[domain]:
        mod_dict[domain]['target_mask'] = torch.ones(mod_dict[domain]['tensor'].shape, dtype=torch.bool, device=device)
    if 'decoder_attention_mask' not in mod_dict[domain]:
        mod_dict[domain]['decoder_attention_mask'] = torch.zeros(mod_dict[domain]['tensor'].shape, dtype=torch.bool, device=device)

    if modality_info[domain]['type'] == 'img':
        mod_dict[domain]['input_mask'][:] = False
        mod_dict[domain]['target_mask'][:] = True

    elif modality_info[domain]['type'] in ['seq', 'seq_token']:
        if eos_id in mod_dict[domain]['tensor']:
            eos_idx = torch.where(mod_dict[domain]['tensor'] == eos_id)[1][0].item()
        else:
            mod_dict[domain]['tensor'][:,0] = eos_id
            eos_idx = 0
        mod_dict[domain]['input_mask'][:,:eos_idx+1] = False
        mod_dict[domain]['input_mask'][:,eos_idx+1:] = True
        mod_dict[domain]['target_mask'][:] = True

    elif modality_info[domain]['type'] in ['seq_emb']:
        # T5 caption has the valid mask saved alongside the embeddings
        mod_dict[domain]['input_mask'] = ~mod_dict[domain]['mask_valid']
        mod_dict[domain]['target_mask'] = torch.ones_like(mod_dict[domain]['mask_valid'])
        mod_dict[domain]['decoder_attention_mask'] = torch.zeros_like(mod_dict[domain]['mask_valid'])
        
    return mod_dict

def custom_text(sample, input_text, eos_token, key, device, text_tokenizer, target_max_len=50, start_token="[S_1]"):
    input_ids = text_tokenizer.encode(input_text).ids
    input_ids = torch.tensor(input_ids).unsqueeze(0)

    target_text = [start_token]
    target_text.extend(["[PAD]"] * (target_max_len - 2))
    target_text.append(eos_token)
    target_text = " ".join(target_text)
    target_ids = text_tokenizer.encode(target_text).ids
    target_ids = torch.tensor(target_ids).unsqueeze(0)

    all_ids = torch.cat([input_ids, target_ids], dim=1)

    input_mask = torch.cat([
        torch.zeros_like(input_ids, dtype=torch.bool), 
        torch.ones_like(target_ids, dtype=torch.bool),
        ], dim=1)

    target_mask = torch.cat([
        torch.ones_like(input_ids, dtype=torch.bool), 
        torch.zeros_like(target_ids, dtype=torch.bool),
        ], dim=1)

    sample[key] = {}
    sample[key]['tensor'] = all_ids.to(device)
    sample[key]['input_mask'] = input_mask.to(device)
    sample[key]['target_mask'] = target_mask.to(device)
    sample[key]['decoder_attention_mask'] = torch.zeros(all_ids.shape, dtype=torch.bool, device=device)

    return sample

def expand_to_batch(mod_dict, batch_size):
    for mod, d in mod_dict.items():
        for k, v in d.items():
            if k in ['tensor', 'input_mask', 'target_mask', 'decoder_attention_mask', 'mask_valid']:
                B = v.shape[0]
                if B == 1:
                    mod_dict[mod][k] = repeat(v, "1 ... -> b ...", b=batch_size)
                elif B != batch_size:
                    raise ValueError(f"Invalid batch size: {B} instead of {batch_size}")
                
    return mod_dict


def build_chained_generation_schedules(
        cond_domains: List[str], 
        target_domains: List[str],
        tokens_per_target: List[int],
        autoregression_schemes: List[str], 
        decoding_steps: List[int], 
        token_decoding_schedules: List[str],
        temps: List[float],
        temp_schedules: List[float],
        cfg_scales: List[float], 
        cfg_schedules: List[str],
        cfg_grow_conditioning: bool = False, 
        modality_info: Optional[dict] = None,
    ):
    """
    Builds a list of chained generation schedules, where each schedule is a tuple of the form:
    (target_modality, schema, number of decoded tokens, temperature, guidance_scale, cfg_cond_domains)

    Args:
        cond_domains: List of conditioning domains
        target_domains: List of target domains
        tokens_per_target: List of number of tokens to decode for each target domain
        autoregression_schemes: List of autoregression schemes for each target domain. maskgit, roar, or autoregressive
        decoding_steps: List of number of maskgit steps for each target domain (if applicable)
        token_decoding_schedules: List of maskgit token schedules for each target domain (if applicable). cosine or linear
        temps: List of starting temperatures for each target domain
        temp_schedules: List of temperature schedules for each target domain. linear, constant, or onex:{min_t}:{power}
        cfg_scales: List of classifier-free guidance scales for each target domain
        cfg_schedules: List of classifier-free guidance schedules for each target domain. constant or cosine
        cfg_grow_conditioning: After every completed modality, add them to classifier-free guidance conditioning
        modality_info: Dictionary with metadata for each modality, optionally used to verify that the schedule is compatible with the modality
    """
    
    # List of {target_modality, schema, number of decoded tokens, temperature, guidance_scale, cfg_cond_domains} dicts
    chained_schedules = []
    
    cond_domains = cond_domains.copy()
    
    for target_idx in range(len(target_domains)):
        
        scheme = autoregression_schemes[target_idx]
        target_domain = target_domains[target_idx]
        ntoks = tokens_per_target[target_idx]
        maskgit_token_schedule_name = token_decoding_schedules[target_idx]
        temp = temps[target_idx]
        temp_schedule_name = temp_schedules[target_idx]
        cfg_scale = cfg_scales[target_idx]
        cfg_schedule_name = cfg_schedules[target_idx]

        # Auto-regressive (caption, detection, ...)
        if scheme == 'autoregressive':
            chained_schedules.append({
                'target_domain': target_domain,
                'scheme': scheme,
                'num_tokens': None,
                'temperature': temp,
                'cfg_scale': cfg_scale,
                'cfg_cond_domains': cond_domains.copy()
            })
            continue

        # Use modality info for (optional) assert if provided
        if modality_info is not None:
            assert modality_info[target_domain]['type'] not in ['seq', 'seq_token'], f'Illegal autoregressive scheme {scheme} for target domain {target_domain}'

        # Token schedule
        if scheme == 'maskgit':
            # MaskGIT token schedule setup
            num_steps = decoding_steps[target_idx]
            if maskgit_token_schedule_name == 'cosine':
                token_schedule = cosine_schedule(num_steps, (ntoks))
            elif maskgit_token_schedule_name == 'linear':
                token_schedule = linear_schedule(num_steps, (ntoks))
            else:
                raise ValueError(f'Illegal MaskGIT token schedule {maskgit_token_schedule_name}')
        elif scheme == 'roar':
            # ROAR token schedule setup (one-by-one, but random order)
            num_steps = decoding_steps[target_idx]            
            token_schedule = linear_schedule(num_steps, ntoks)
        else:
            raise ValueError(f'Illegal decoding scheme {scheme}')
        
        # Temperature schedule
        if temp_schedule_name == 'linear':
            temp_schedule = linear_temp_schedule(temp, token_schedule)
        elif temp_schedule_name == 'constant':
            temp_schedule = temp * np.ones(num_steps)
        elif 'onex' in temp_schedule_name:
            # onex temperature schedule has to be formatted like onex:{min_t}:{power}
            min_t, power = [float(f) for f in temp_schedule_name.split(':')[1:]]
            temp_schedule = onex_temp_schedule(max_t=temp, min_t=min_t, token_schedule=token_schedule, power=power)
        else:
            raise ValueError(f'Illegal temperature schedule {temp_schedule_name}')

        # Classifier-free guidance scale schedule
        if cfg_schedule_name == 'constant':
            if isinstance(cfg_scale, float):
                cfg_schedule = cfg_scale * np.ones(num_steps)
            elif isinstance(cfg_scale, list):
                cfg_schedule = np.array(cfg_scale) * np.ones(num_steps).reshape(-1, 1)
        elif cfg_schedule_name == 'cosine':
            raise NotImplementedError()
        else:
            raise ValueError(f'Illegal guidance schedule {cfg_schedule_name}')

        # Concatenate schedule for this modality with previous ones
        schedule = [
            {
                'target_domain': target_domain,
                'scheme': scheme,
                'num_tokens': tok,
                'temperature': temp,
                'cfg_scale': cfg,
                'cfg_cond_domains': cond_domains.copy()
            }
            for tok, temp, cfg in zip(token_schedule, temp_schedule, cfg_schedule)
        ]
        chained_schedules.extend(schedule)
        
        # Optionally add this new modality to the ones affected by classifier-free guidance
        if cfg_grow_conditioning:
            cond_domains.append(target_domain)
            
    return chained_schedules


class GenerationSampler(nn.Module):
    """Sampler that wraps a trained 4M model for generation use cases.
    Implements standard autoregressive, MaskGIT, and ROAR generation schemes with chaining and weighted guidance."""

    def __init__(self, model):
        super().__init__()
        self.model = model


    def top_k_top_p_filtering(self, logits, top_k=0.0, top_p=0.0):
        # Compatible with batching
        # From https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        if top_k > 0.0:
            if isinstance(top_k, int):
                k = min(top_k, logits.shape[-1])
            elif isinstance(top_k, float):
                k = min(int(top_k * logits.shape[-1]), logits.shape[-1])
            else:
                raise ValueError(f"Invalid value for top_k: {top_k}")

            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, dim=1, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cum_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            restore_indices = torch.argsort(sorted_indices, dim=-1)
            indices_to_remove = torch.gather(sorted_indices_to_remove, dim=-1, index=restore_indices)
            logits[indices_to_remove] = float("-inf")

        return logits

    def sample_tokens(self, logits, temperature=1.0, top_k=0.0, top_p=0.0):
        if np.isclose(temperature, 0, atol=1e-10):
            samples = torch.argmax(logits, dim=-1)
            # Since argmax is used, all sampled_probs will be 1 as we're selecting the max probability
            sampled_probs = torch.ones_like(samples, dtype=torch.float32)
        else:
            filtered_logits = self.top_k_top_p_filtering(logits, top_k, top_p)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            samples = torch.multinomial(probs, 1)[:, 0]
            sampled_probs = probs[torch.arange(len(samples)), samples]
        return samples, sampled_probs
    
    def sample_tokens_batched(self, logits, temperature=1.0, top_k=0.0, top_p=0.0):
        if logits.ndim > 2:
            B, N = logits.shape[0], logits.shape[1]
            logits = rearrange(logits, 'b n v -> (b n) v')
            samples, sampled_probs = self.sample_tokens(logits, temperature, top_k, top_p)
            samples = rearrange(samples, '(b n) -> b n', b=B, n=N)
            sampled_probs = rearrange(sampled_probs, '(b n) -> b n', b=B, n=N)
            return samples, sampled_probs
        else:
            return self.sample_tokens(logits, temperature, top_k, top_p)

    def select_tokens(self, logits, num_select, temperature=1.0, top_k=0.0, top_p=0.0, return_all_samples=False):
        samples, sampled_probs = self.sample_tokens(logits, temperature, top_k, top_p)
        top_indices = torch.topk(sampled_probs, num_select)[1]
        top_samples = samples[top_indices]
        if return_all_samples:
            return top_samples, top_indices, samples
        else:
            return top_samples, top_indices
        
    def select_tokens_batched(self, logits, num_select, temperature=1.0, top_k=0.0, top_p=0.0, return_all_samples=False):
            if logits.ndim > 2:
                samples, sampled_probs = self.sample_tokens_batched(logits, temperature, top_k, top_p) # both of shape (B, N)
                top_indices = torch.topk(sampled_probs, num_select, dim=-1)[1]
                # Need to switch to gather instead of indexing here
                top_samples = torch.gather(samples, dim=-1, index=top_indices)
                if return_all_samples:
                    return top_samples, top_indices, samples
                else:
                    return top_samples, top_indices
            else:
                return self.sample_tokens(logits, num_select, temperature, top_k, top_p, return_all_samples)


    def forward_mask_encoder_generation(self, encoder_mod_dict):
        """Modification of forward_mask_encoder adapted for generation, with support for batching
        """
        # Form input
        B = list(encoder_mod_dict.values())[0]['tensor'].shape[0]

        encoder_tokens_all, emb_all, encoder_mask_all, mod_mask_all = self.model.cat_encoder_tensors(encoder_mod_dict)
        # Take max num encoder of tokens (although assuming it's the same everywhere would be better)
        num_encoder_tokens = (~encoder_mask_all.reshape(B, -1)).sum(dim=1).max()

        # Add arange multiplied by small constant to mask so they get sorted in a deterministic way
        mask_arange = torch.arange(encoder_mask_all.shape[1], device=encoder_mask_all.device).unsqueeze(0) * 1e-6
        ids_shuffle = torch.argsort(encoder_mask_all + mask_arange, dim=1)
        # ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_encoder_tokens]

        encoder_tokens = torch.gather(encoder_tokens_all, dim=1,
                                      index=repeat(ids_keep, "b n -> b n d", d=encoder_tokens_all.shape[2]))
        encoder_emb = torch.gather(emb_all, dim=1, index=repeat(ids_keep, "b n -> b n d", d=emb_all.shape[2]))
        encoder_mask = torch.gather(encoder_mask_all, dim=1, index=ids_keep)
        mod_mask = torch.gather(mod_mask_all, dim=1, index=ids_keep)

        if self.model.num_register_tokens > 0:
            prompt_tokens = repeat(self.prompt_tokens, '() n d -> b n d', b=B)
            # We add prompt tokens at the beginning of the sequence
            encoder_tokens = torch.cat([prompt_tokens, encoder_tokens], dim=1)
            encoder_emb = torch.cat([torch.zeros_like(prompt_tokens), encoder_emb], dim=1)
            encoder_mask = torch.cat([torch.zeros((B, prompt_tokens.shape[1]), dtype=torch.bool, device=encoder_mask.device), encoder_mask], dim=1)
            mod_mask = torch.cat([torch.full((B, prompt_tokens.shape[1]), -1, dtype=torch.int16, device=mod_mask.device), mod_mask], dim=1)

        encoder_tokens[encoder_mask] = 0.
        encoder_emb[encoder_mask] = 0.
        mod_mask[encoder_mask] = -1
        # Mask could be of shape 'b n1 n2' but not needed for masked_fill
        # This means this mask can then be re-used for decoder cross-attention
        encoder_mask = rearrange(encoder_mask, 'b n2 -> b 1 n2')

        return encoder_tokens, encoder_emb, encoder_mask, mod_mask


    def forward_mask_decoder_maskgit(self, mod_dict, target_mod, seed=None):
        """Modification of forward_mask_decoder for MaskGIT generation, with support for batching
        """
        if seed is not None:
            torch.manual_seed(seed)
        d = mod_dict[target_mod]
        decoder_tokens_all = torch.zeros_like(d['x']) + self.model.mask_token
        emb_all = d['emb']
        decoder_mask_all = d['target_mask']
        B = decoder_tokens_all.shape[0] # Get batch size
        mod_mask_all = torch.full_like(d['ids'], self.model.modality_info[target_mod]['id'], dtype=torch.int16)
        mod_pos_all = torch.arange(d['x'].shape[1], device=d['x'].device).unsqueeze(0)
        mod_pos_all = repeat(mod_pos_all, '1 n -> b n', b=B) # Added: Expansion for batching
        num_decoder_tokens = (~decoder_mask_all[0]).sum()  # Adapted for batching / Assumes num_decoder_tokens is the same across the batch


        # Add arange multiplied by small constant to mask so they get sorted in a deterministic way
        mask_arange = torch.arange(decoder_mask_all.shape[1], device=decoder_mask_all.device).unsqueeze(0) * 1e-6
        ids_shuffle = torch.argsort(decoder_mask_all + mask_arange, dim=1)
        # ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_decoder_tokens]

        decoder_tokens = torch.gather(decoder_tokens_all, dim=1, index=repeat(ids_keep, "b n -> b n d", d=decoder_tokens_all.shape[2]))
        decoder_emb = torch.gather(emb_all, dim=1, index=repeat(ids_keep, "b n -> b n d", d=emb_all.shape[2]))
        decoder_mask = torch.gather(decoder_mask_all, dim=1, index=ids_keep)
        mod_mask = torch.gather(mod_mask_all, dim=1, index=ids_keep)
        mod_pos = torch.gather(mod_pos_all, dim=1, index=ids_keep)

        decoder_tokens[decoder_mask] = 0.
        decoder_emb[decoder_mask] = 0.
        mod_mask[decoder_mask] = -1

        return decoder_tokens, decoder_emb, decoder_mask, mod_mask, mod_pos

    def forward_mask_decoder_roar(self, mod_dict, target_mod, num_select, seed=None):
        """Modification of forward_mask_decoder for ROAR generation, with support for batching
        """
        if seed is not None:
            torch.manual_seed(seed)
        d = mod_dict[target_mod]
        decoder_tokens_all = torch.zeros_like(d['x']) + self.model.mask_token
        emb_all = d['emb']
        decoder_mask_all = d['target_mask']
        B = decoder_tokens_all.shape[0] # Get batch size
        mod_mask_all = torch.full_like(d['ids'], self.model.modality_info[target_mod]['id'], dtype=torch.int16)
        mod_pos_all = torch.arange(d['x'].shape[1], device=d['x'].device).unsqueeze(0)
        mod_pos_all = repeat(mod_pos_all, '1 n -> b n', b=B) # Added: Expansion for batching
        # Only keep the first num_select tokens
        num_decoder_tokens = min(num_select, (~decoder_mask_all[0]).sum())  # Adapted for batching / Assumes num_decoder_tokens is the same across the batch
 
        # Add a small random number to the mask so they get sorted in a random way, but keeping the masked tokens first
        mask_rand = torch.rand(decoder_mask_all.shape[1], device=decoder_mask_all.device).unsqueeze(0) * 1e-6
        ids_shuffle = torch.argsort(decoder_mask_all + mask_rand, dim=1)
        # ids_restore = torch.argsort(ids_shuffle, dim=1)
        # Only keep the first num_select_tokens
        ids_keep = ids_shuffle[:, :num_decoder_tokens]

        decoder_tokens = torch.gather(decoder_tokens_all, dim=1, index=repeat(ids_keep, "b n -> b n d", d=decoder_tokens_all.shape[2]))
        decoder_emb = torch.gather(emb_all, dim=1, index=repeat(ids_keep, "b n -> b n d", d=emb_all.shape[2]))
        decoder_mask = torch.gather(decoder_mask_all, dim=1, index=ids_keep)
        mod_mask = torch.gather(mod_mask_all, dim=1, index=ids_keep)
        mod_pos = torch.gather(mod_pos_all, dim=1, index=ids_keep)

        decoder_tokens[decoder_mask] = 0.
        decoder_emb[decoder_mask] = 0.
        mod_mask[decoder_mask] = -1

        return decoder_tokens, decoder_emb, decoder_mask, mod_mask, mod_pos

    def forward_mask_decoder_autoregressive(self, mod_dict, target_mod, seed=None):
        # Adapted for batching
        if seed is not None:
            torch.manual_seed(seed)
        # This is the concatenation part
        d = mod_dict[target_mod]
        decoder_ids_all = d['ids']
        emb_all = d['emb']
        decoder_mask_all = d['target_mask']
        B = decoder_ids_all.shape[0] # Get batch size
        mod_mask_all = torch.full_like(d['ids'], self.model.modality_info[target_mod]['id'], dtype=torch.int16)
        mod_pos_all = torch.arange(d['x'].shape[1], device=d['x'].device).unsqueeze(0)
        mod_pos_all = repeat(mod_pos_all, '1 n -> b n', b=B) 
        num_decoder_tokens = (~decoder_mask_all[0]).sum() # Adapted for batching, but assumes num_decoder_tokens is the same across the batch
 
        # Add arange multiplied by small constant to mask so they get sorted in a deterministic way
        mask_arange = torch.arange(decoder_mask_all.shape[1], device=decoder_mask_all.device).unsqueeze(0) * 1e-6
        ids_shuffle = torch.argsort(decoder_mask_all + mask_arange, dim=1)
        # ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_decoder_tokens]

        # Same as in forward_mask_decoder
        decoder_ids = torch.gather(decoder_ids_all, dim=1, index=ids_keep)
        decoder_emb = torch.gather(emb_all, dim=1, index=repeat(ids_keep, "b n -> b n d", d=emb_all.shape[2]))
        decoder_mask = torch.gather(decoder_mask_all, dim=1, index=ids_keep)
        mod_mask = torch.gather(mod_mask_all, dim=1, index=ids_keep)
        mod_pos = torch.gather(mod_pos_all, dim=1, index=ids_keep)

        decoder_ids[decoder_mask] = 0
        decoder_emb[decoder_mask] = 0.
        mod_mask[decoder_mask] = -1

        return decoder_ids, decoder_emb, decoder_mask, mod_mask, mod_pos

    def merge_sequences(self, mod_dict, pred_ids, target_mod, text_tokenizer, default_sentinel="[S_1]"):
        device = mod_dict[target_mod]['tensor'].device
        # Get input ids
        input_ids = mod_dict[target_mod]['tensor'].squeeze().detach().cpu()
        input_ids = input_ids[mod_dict[target_mod]['input_mask'].squeeze().detach().cpu() == 0]
        input_ids = input_ids.tolist()

        if len(input_ids) == 0:
            input_ids = [text_tokenizer.get_vocab()[default_sentinel]]

        # Get predicted ids
        pred_ids = pred_ids.squeeze().detach().cpu().tolist()
        if isinstance(pred_ids, int):
            pred_ids = [pred_ids]

        # Get sentinel ids using the tokenizer
        sentinel_ids = set(get_sentinel_to_id_mapping(text_tokenizer).values())
        # Perform merging
        merged_ids = merge_span_masking(input_ids, pred_ids, sentinel_ids)
        merged_ids = torch.tensor(merged_ids).unsqueeze(0)
        # Create new dict
        new_input_mask = torch.zeros_like(merged_ids, dtype=torch.bool)
        new_target_mask = torch.ones_like(merged_ids, dtype=torch.bool)
        new_dict = {'tensor': merged_ids.to(device),
                    'input_mask': new_input_mask.to(device),
                    'target_mask': new_target_mask.to(device)}
        new_dict['decoder_attention_mask'] = torch.zeros_like(new_target_mask, dtype=torch.bool)

        mod_dict[target_mod] = new_dict
        return mod_dict
    
    def merge_sequences_batched(self, mod_dict, pred_ids, target_mod, text_tokenizer, default_sentinel="[S_1]"):
        # Unbatches and calls merge sequence per batch, then regroups it into a batch
        
        pad_id = text_tokenizer.token_to_id("[PAD]")
        
        B = mod_dict[target_mod]['tensor'].shape[0]
        device = mod_dict[target_mod]['tensor'].device

        tensors = torch.split(mod_dict[target_mod]['tensor'], 1)
        input_masks = torch.split(mod_dict[target_mod]['input_mask'], 1)
        pred_ids = torch.split(pred_ids, 1)

        input_dicts = []
        for t, im in zip(tensors, input_masks):
            d = {target_mod: {'tensor': t, 'input_mask': im}}
            input_dicts.append(d)

        merged_tensors = []
        merged_input_masks = []
        merged_target_masks = []
        merged_seq_lens = []
        for input_d, pi in zip(input_dicts, pred_ids):
            # Output of merge_sequences is mod_dict with modified target mod
            merged_d = self.merge_sequences(input_d, pi, target_mod, text_tokenizer, default_sentinel)[target_mod]
            merged_tensors.append(merged_d['tensor'])
            merged_input_masks.append(merged_d['input_mask'])
            merged_target_masks.append(merged_d['input_mask'])
            merged_seq_lens.append(merged_d['tensor'].shape[1])


        max_seq_len = max(merged_seq_lens)

        for i in range(len(merged_tensors)):
            # Right pad all tensors
            p1d = (0, max_seq_len - merged_seq_lens[i])
            merged_tensors[i] = F.pad(merged_tensors[i], p1d, "constant",pad_id)
            merged_input_masks[i] = F.pad(merged_input_masks[i], p1d, "constant", True)
            merged_target_masks[i] = F.pad(merged_target_masks[i], p1d, "constant", True)

        new_dict = {'tensor': torch.cat(merged_tensors, dim=0).to(device),
                    'input_mask': torch.cat(merged_input_masks, dim=0).to(device),
                    'target_mask': torch.cat(merged_target_masks, dim=0).to(device)}
        new_dict['decoder_attention_mask'] = torch.zeros_like(new_dict['target_mask'], dtype=torch.bool)

        mod_dict[target_mod] = new_dict
        return mod_dict

    def forward_enc_dec_maskgit_batched(self, mod_dict, target_mod, seed=None):
        # Encoder
        encoder_mod_dict = {mod: self.model.encoder_embeddings[mod](d)
                            for mod, d in mod_dict.items()
                            if mod in self.model.encoder_embeddings}
        encoder_tokens, encoder_emb, encoder_mask, encoder_mod_mask = self.forward_mask_encoder_generation(encoder_mod_dict)
        x = encoder_tokens + encoder_emb
        x = self.model.forward_encoder(x, encoder_mask)

        # Decoder
        context = self.model.decoder_proj_context(x) + encoder_emb
        decoder_mod_dict = {target_mod: self.model.decoder_embeddings[target_mod].forward_embed(mod_dict[target_mod])}
        decoder_tokens, decoder_emb, decoder_mask, decoder_mod_mask, mod_pos = self.forward_mask_decoder_maskgit(decoder_mod_dict, target_mod, seed=seed)
        y = decoder_tokens + decoder_emb
        y = self.model.forward_decoder(y, context, encoder_mask, None)
        B, N, D = y.shape
        logits = self.model.forward_logits(y, decoder_mod_dict, decoder_mod_mask)[target_mod]
        logits = logits.reshape(B, N, -1)


        return logits, mod_pos

    def maskgit_step_batched(self, mod_dict, target_mod, num_select, temperature, top_k, top_p, seed=None):        
        logits, mod_pos = self.forward_enc_dec_maskgit_batched(mod_dict, target_mod, seed=seed)

        # MaskGIT sampling
        top_samples, top_indices = self.select_tokens_batched(logits, num_select,
                                                      temperature=temperature, top_k=top_k, top_p=top_p)
        # Update mod dict
        # We rely on gather / scatter for batched operations
        top_pos = torch.gather(mod_pos, -1, top_indices) # (B, num_select)
        mod_dict[target_mod]['tensor'] = torch.scatter(mod_dict[target_mod]['tensor'], -1, top_pos, top_samples)
        mod_dict[target_mod]['input_mask'] = torch.scatter(mod_dict[target_mod]['input_mask'], -1, top_pos, torch.zeros_like(top_samples, dtype=torch.bool))
        mod_dict[target_mod]['target_mask'] = torch.scatter(mod_dict[target_mod]['target_mask'], -1, top_pos, torch.ones_like(top_samples, dtype=torch.bool))

        return mod_dict

    def guided_maskgit_step_batched(self, mod_dict, target_mod, num_select, temperature, top_k, top_p, 
                                    conditioning=[], guidance_scale=1.0, seed=None, write_all_predictions=False):

        ### 1 - First pass, with conditioning
        logits_cond, _ = self.forward_enc_dec_maskgit_batched(mod_dict, target_mod, seed=seed)

        ### 2 - Second pass, without conditioning
        mod_dict_uncond = copy.deepcopy(mod_dict)
        for mod in conditioning:
            if self.model.modality_info[mod]['type'] in ['seq', 'seq_token']:
                mod_dict_uncond = empty_seq_modality(mod_dict_uncond, mod)
            elif self.model.modality_info[mod]['type'] in ['seq_emb']: 
                mod_dict_uncond = empty_seq_emb_modality(mod_dict_uncond, mod)
            else:
                mod_dict_uncond = empty_img_modality(mod_dict_uncond, mod)

        logits_uncond, mod_pos = self.forward_enc_dec_maskgit_batched(mod_dict_uncond, target_mod, seed=seed)

        ### 3 - Classifier-free guidance
        logits = logits_uncond + (logits_cond - logits_uncond) * guidance_scale

        ### 4 - MaskGIT sampling
        top_samples, top_indices, all_samples = self.select_tokens_batched(
            logits, num_select,
            temperature=temperature, top_k=top_k, top_p=top_p, 
            return_all_samples=True
        )

        ### 5 - Update mod dict
        # We rely on gather / scatter for batched operations
        top_pos = torch.gather(mod_pos, -1, top_indices) # (B, num_select)
        if write_all_predictions:
            mod_dict[target_mod]['tensor'][:, mod_pos] = all_samples
        else:
            mod_dict[target_mod]['tensor'] = torch.scatter(mod_dict[target_mod]['tensor'], -1, top_pos, top_samples)
        mod_dict[target_mod]['input_mask'] = torch.scatter(mod_dict[target_mod]['input_mask'], -1, top_pos, torch.zeros_like(top_samples, dtype=torch.bool))
        mod_dict[target_mod]['target_mask'] = torch.scatter(mod_dict[target_mod]['target_mask'], -1, top_pos, torch.ones_like(top_samples, dtype=torch.bool))

        return mod_dict

    def multi_guided_maskgit_step_batched(self, uncond_dict, cond_dicts, cond_weights, target_mod, num_select, 
                                          temperature, top_k, top_p, seed=None, write_all_predictions=False):

        ### 1 - Conditional forward passes (one for each guided condition)
        logits_cond_all = []
        for cond_dict in cond_dicts:
            logits_cond_i, _ = self.forward_enc_dec_maskgit_batched(cond_dict, target_mod, seed=seed)
            logits_cond_all.append(logits_cond_i)
        
        ### 2 - Unconditional forward pass
        logits_uncond, mod_pos = self.forward_enc_dec_maskgit_batched(uncond_dict, target_mod, seed=seed)

        ### 3 Conjunction of multiple conditions: l_uncond + sum_i{w_i * (l_cond_i - l_uncond)}
        # See https://arxiv.org/abs/2206.01714
        logits = logits_uncond + torch.stack([w * (logits_cond - logits_uncond) for w, logits_cond in zip(cond_weights, logits_cond_all)]).sum(dim=0)

        ### 4 - MaskGIT sampling
        top_samples, top_indices, all_samples = self.select_tokens_batched(
            logits, num_select,
            temperature=temperature, top_k=top_k, top_p=top_p, 
            return_all_samples=True
        )
        
        ### 5 - Update mod dict with newly generated tokens
        # We rely on gather / scatter for batched operations
        top_pos = torch.gather(mod_pos, -1, top_indices) # (B, num_select)
        if write_all_predictions:
            uncond_dict[target_mod]['tensor'][:, mod_pos] = all_samples
        else:
            uncond_dict[target_mod]['tensor'] = torch.scatter(uncond_dict[target_mod]['tensor'], -1, top_pos, top_samples)
        uncond_dict[target_mod]['input_mask'] = torch.scatter(uncond_dict[target_mod]['input_mask'], -1, top_pos, torch.zeros_like(top_samples, dtype=torch.bool))
        uncond_dict[target_mod]['target_mask'] = torch.scatter(uncond_dict[target_mod]['target_mask'], -1, top_pos, torch.ones_like(top_samples, dtype=torch.bool))
        # Update conditioning dicts
        for i in range(len(cond_dicts)):
            cond_dicts[i][target_mod]['tensor'] = torch.scatter(cond_dicts[i][target_mod]['tensor'], -1, top_pos, top_samples)
            cond_dicts[i][target_mod]['input_mask'] = torch.scatter(cond_dicts[i][target_mod]['input_mask'], -1, top_pos, torch.zeros_like(top_samples, dtype=torch.bool))
            cond_dicts[i][target_mod]['target_mask'] = torch.scatter(cond_dicts[i][target_mod]['target_mask'], -1, top_pos, torch.ones_like(top_samples, dtype=torch.bool))

        return uncond_dict, cond_dicts

    def forward_enc_dec_roar_batched(self, mod_dict, target_mod, num_select, seed=None):
        # Encoder
        encoder_mod_dict = {mod: self.model.encoder_embeddings[mod](d)
                            for mod, d in mod_dict.items()
                            if mod in self.model.encoder_embeddings}
        encoder_tokens, encoder_emb, encoder_mask, encoder_mod_mask = self.forward_mask_encoder_generation(encoder_mod_dict)
        x = encoder_tokens + encoder_emb
        x = self.model.forward_encoder(x, encoder_mask)

        # Decoder
        context = self.model.decoder_proj_context(x) + encoder_emb
        decoder_mod_dict = {target_mod: self.model.decoder_embeddings[target_mod].forward_embed(mod_dict[target_mod])}
        decoder_tokens, decoder_emb, decoder_mask, decoder_mod_mask, mod_pos = self.forward_mask_decoder_roar(decoder_mod_dict, target_mod, num_select, seed=seed)
        y = decoder_tokens + decoder_emb
        y = self.model.forward_decoder(y, context, encoder_mask, None)
        B, N, D = y.shape
        logits = self.model.forward_logits(y, decoder_mod_dict, decoder_mod_mask)[target_mod]
        logits = logits.reshape(B, N, -1)

        return logits, mod_pos

    def roar_step_batched(self, mod_dict, target_mod, num_select, temperature, top_k, top_p, seed=None):
        """ROAR = Random Order Autoregression"""

        logits, mod_pos = self.forward_enc_dec_roar_batched(mod_dict, target_mod, num_select, seed=seed)

        # Simple sampling
        samples, sampled_probs = self.sample_tokens_batched(logits, temperature, top_k=top_k, top_p=top_p)

        # Update mod dict
        # We rely on scatter for batched operations
        select_pos = mod_pos
        mod_dict[target_mod]['tensor'] = torch.scatter(mod_dict[target_mod]['tensor'], -1, select_pos, samples)
        mod_dict[target_mod]['input_mask'] = torch.scatter(mod_dict[target_mod]['input_mask'], -1, select_pos, torch.zeros_like(samples, dtype=torch.bool))
        mod_dict[target_mod]['target_mask'] = torch.scatter(mod_dict[target_mod]['target_mask'], -1, select_pos, torch.ones_like(samples, dtype=torch.bool))

        return mod_dict

    def guided_roar_step_batched(self, mod_dict, target_mod, num_select, temperature, top_k, top_p, 
                                 conditioning=[], guidance_scale=1.0, seed=None):
        """ROAR = Random Order Autoregression"""

        ### 1 - First pass, with conditioning
        logits_cond, _ = self.forward_enc_dec_roar_batched(mod_dict, target_mod, num_select, seed=seed)

        ### 2 - Second pass, without conditioning
        mod_dict_uncond = copy.deepcopy(mod_dict)
        for mod in conditioning:
            if self.model.modality_info[mod]['type'] in ['seq', 'seq_token']:
                mod_dict_uncond = empty_seq_modality(mod_dict_uncond, mod)
            elif self.model.modality_info[mod]['type'] in ['seq_emb']: 
                mod_dict_uncond = empty_seq_emb_modality(mod_dict_uncond, mod)
            else:
                mod_dict_uncond = empty_img_modality(mod_dict_uncond, mod)

        logits_uncond, mod_pos = self.forward_enc_dec_roar_batched(mod_dict_uncond, target_mod, num_select, seed=seed)

        ### 3 - Classifier-free guidance
        logits = logits_uncond + (logits_cond - logits_uncond) * guidance_scale

        ### 4 - Simple sampling
        samples, sampled_probs = self.sample_tokens_batched(logits, temperature, top_k=top_k, top_p=top_p)

        ### 5 - Update mod dict
        # We rely on gather / scatter for batched operations
        select_pos = mod_pos
        mod_dict[target_mod]['tensor'] = torch.scatter(mod_dict[target_mod]['tensor'], -1, select_pos, samples)
        mod_dict[target_mod]['input_mask'] = torch.scatter(mod_dict[target_mod]['input_mask'], -1, select_pos, torch.zeros_like(samples, dtype=torch.bool))
        mod_dict[target_mod]['target_mask'] = torch.scatter(mod_dict[target_mod]['target_mask'], -1, select_pos, torch.ones_like(samples, dtype=torch.bool))

        return mod_dict

    def multi_guided_roar_step_batched(self, uncond_dict, cond_dicts, cond_weights, target_mod, 
                                       num_select, temperature, top_k, top_p, seed=None):

        ### 1 - Conditional forward passes (one for each guided condition)
        logits_cond_all = []
        for cond_dict in cond_dicts:
            logits_cond_i, _ = self.forward_enc_dec_roar_batched(cond_dict, target_mod, num_select, seed=seed)
            logits_cond_all.append(logits_cond_i)
        
        ### 2 - Unconditional forward pass
        logits_uncond, mod_pos = self.forward_enc_dec_roar_batched(uncond_dict, target_mod, num_select, seed=seed)

        ### 3 Conjunction of multiple conditions: l_uncond + sum_i{w_i * (l_cond_i - l_uncond)}
        # See https://arxiv.org/abs/2206.01714
        logits = logits_uncond + torch.stack([w * (logits_cond - logits_uncond) for w, logits_cond in zip(cond_weights, logits_cond_all)]).sum(dim=0)

        ### 4 - Simple sampling
        samples, sampled_probs = self.sample_tokens_batched(logits, temperature, top_k=top_k, top_p=top_p)
        
        ### 5 - Update mod dict
        # We rely on gather / scatter for batched operations
        select_pos = mod_pos
        uncond_dict[target_mod]['tensor'] = torch.scatter(uncond_dict[target_mod]['tensor'], -1, select_pos, samples)
        uncond_dict[target_mod]['input_mask'] = torch.scatter(uncond_dict[target_mod]['input_mask'], -1, select_pos, torch.zeros_like(samples, dtype=torch.bool))
        uncond_dict[target_mod]['target_mask'] = torch.scatter(uncond_dict[target_mod]['target_mask'], -1, select_pos, torch.ones_like(samples, dtype=torch.bool))
        # Update conditioning dicts
        for i in range(len(cond_dicts)):
            cond_dicts[i][target_mod]['tensor'] = torch.scatter(cond_dicts[i][target_mod]['tensor'], -1, select_pos, samples)
            cond_dicts[i][target_mod]['input_mask'] = torch.scatter(cond_dicts[i][target_mod]['input_mask'], -1, select_pos, torch.ones_like(samples, dtype=torch.bool))
            cond_dicts[i][target_mod]['target_mask'] = torch.scatter(cond_dicts[i][target_mod]['target_mask'], -1, select_pos, torch.zeros_like(samples, dtype=torch.bool))
        
        return uncond_dict, cond_dicts

    def autoregressive_step_batched(self, mod_dict, target_mod, temperature, top_k: Union[float, int], top_p: float,
                                    use_eos=True, eos_token=None, start_tokens=None, text_tokenizer=None, seed=None):

        # Encoder
        encoder_mod_dict = {mod: self.model.encoder_embeddings[mod](d)
                            for mod, d in mod_dict.items()
                            if mod in self.model.encoder_embeddings}
        encoder_tokens, encoder_emb, encoder_mask, encoder_mod_mask = self.forward_mask_encoder_generation(encoder_mod_dict)
        x = encoder_tokens + encoder_emb
        x = self.model.forward_encoder(x, encoder_mask) # B, N, D

        # Get batch size
        B = x.shape[0]

        # Decoder
        context = self.model.decoder_proj_context(x) + encoder_emb
        decoder_mod_dict = {target_mod: self.model.decoder_embeddings[target_mod].forward_embed(mod_dict[target_mod])}
        decoder_ids, decoder_emb, decoder_mask, decoder_mod_mask, mod_pos = self.forward_mask_decoder_autoregressive(decoder_mod_dict, target_mod, seed=seed)
        device = decoder_ids.device
        seq_len = self.model.modality_info[target_mod]['max_tokens']

        if use_eos and eos_token is None:
            # The eos_token is the final sentinel token provided
            eos_token = decoder_ids[0][decoder_mask[0] == 0][-1] # Assumes the EOS token is the same for all
        if use_eos:
            eos_token = eos_token.to(device)

        # If no start_tokens, just use the beginning of the actual target (i.e., a sentinel token)
        out = decoder_ids[:, :1] if start_tokens is None else start_tokens.to(device)
        # Set decoder_tokens to None, we do not use them for decoding
        decoder_ids = None

        # If all samples of the batch have eos, return early
        if use_eos and (out == eos_token).any(dim=-1).all():
            return out

        y_emb = decoder_emb[:, :seq_len]
        seq_len = y_emb.shape[1]

        # Auto-regressive decoding and sampling
        for i in range(seq_len):
            cur_len = out.shape[1]
            # Convert ids into word embeddings and add corresponding posembs + modemb
            y = self.model.decoder_embeddings[target_mod].token_emb(out) + y_emb[:, :cur_len]
            # Build causal mask
            causal_mask = torch.ones((cur_len, cur_len), dtype=torch.bool, device=y.device).triu(1)
            causal_mask = repeat(causal_mask, "n1 n2 -> b n1 n2", b=B)
            
            y = self.model.forward_decoder(y, context, encoder_mask, causal_mask)
            logits = self.model.forward_logits(y, decoder_mod_dict, decoder_mod_mask[:, :cur_len])[target_mod]
            logits = rearrange(logits, "(b n) d -> b n d", b=B, n=cur_len)
            last_logits = logits[:, -1]

            # Sample token for the newly generated logit
            if np.isclose(temperature, 0, atol=1e-10):
                sample = torch.argmax(last_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = self.top_k_top_p_filtering(last_logits, top_k, top_p)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim=-1)

            if use_eos and (out == eos_token).any(dim=-1).all():
                break

        mod_dict = self.merge_sequences_batched(mod_dict, out, target_mod, text_tokenizer)

        return mod_dict

    def guided_autoregressive_step_batched(self, mod_dict, target_mod, temperature, top_k: Union[float, int], top_p: float,
                                           use_eos=True, eos_token=None, start_tokens=None, text_tokenizer=None, 
                                           conditioning=[], guidance_scale=1.0, seed=None):

         ### 1 - Encoder forward pass, with conditioning
        
        # Encoder
        encoder_mod_dict = {mod: self.model.encoder_embeddings[mod](d)
                            for mod, d in mod_dict.items()
                            if mod in self.model.encoder_embeddings}
        encoder_tokens, encoder_emb, encoder_mask_cond, encoder_mod_mask = self.forward_mask_encoder_generation(encoder_mod_dict)
        x = encoder_tokens + encoder_emb
        x = self.model.forward_encoder(x, encoder_mask_cond) # B, N, D

        # Get batch size
        B = x.shape[0]

        # Decoder
        context_cond = self.model.decoder_proj_context(x) + encoder_emb
        decoder_mod_dict_cond = {target_mod: self.model.decoder_embeddings[target_mod].forward_embed(mod_dict[target_mod])}
        decoder_ids, decoder_emb, decoder_mask, decoder_mod_mask_cond, mod_pos = self.forward_mask_decoder_autoregressive(decoder_mod_dict_cond, target_mod, seed=seed)
        device = decoder_ids.device
        seq_len = self.model.modality_info[target_mod]['max_tokens']


        ### 2 - Encoder forward pass, without conditioning

        mod_dict_uncond = copy.deepcopy(mod_dict)
        for mod in conditioning:
            if self.model.modality_info[mod]['type'] in ['seq', 'seq_token']:
                mod_dict_uncond = empty_seq_modality(mod_dict_uncond, mod)
            elif self.model.modality_info[mod]['type'] in ['seq_emb']: 
                mod_dict_uncond = empty_seq_emb_modality(mod_dict_uncond, mod)
            else:
                mod_dict_uncond = empty_img_modality(mod_dict_uncond, mod)

        # Encoder
        encoder_mod_dict = {mod: self.model.encoder_embeddings[mod](d)
                            for mod, d in mod_dict_uncond.items()
                            if mod in self.model.encoder_embeddings}
        encoder_tokens, encoder_emb, encoder_mask_uncond, encoder_mod_mask = self.forward_mask_encoder_generation(encoder_mod_dict)
        x = encoder_tokens + encoder_emb
        x = self.model.forward_encoder(x, encoder_mask_uncond) # B, N, D

        # Decoder
        context_uncond = self.model.decoder_proj_context(x) + encoder_emb
        decoder_mod_dict_uncond = {target_mod: self.model.decoder_embeddings[target_mod].forward_embed(mod_dict[target_mod])}
        decoder_ids, decoder_emb, decoder_mask, decoder_mod_mask_uncond, mod_pos = self.forward_mask_decoder_autoregressive(decoder_mod_dict_uncond, target_mod, seed=seed)


        if use_eos and eos_token is None:
            # The eos_token is the final sentinel token provided
            eos_token = decoder_ids[0][decoder_mask[0] == 0][-1] # Assumes the EOS token is the same for all
        if use_eos:
            eos_token = eos_token.to(device)

        # If no start_tokens, just use the beginning of the actual target (i.e., a sentinel token)
        out = decoder_ids[:, :1] if start_tokens is None else start_tokens.to(device)
        # Set decoder_tokens to None, we do not use them for decoding
        decoder_ids = None

        # If all samples of the batch have eos, return early
        if use_eos and (out == eos_token).any(dim=-1).all():
            return out

        y_emb = decoder_emb[:, :seq_len]
        seq_len = y_emb.shape[1]

        ### 3 -  Auto-regressive decoding and sampling
        for i in range(seq_len):
            cur_len = out.shape[1]
            # Convert ids into word embeddings and add corresponding posembs + modemb
            y = self.model.decoder_embeddings[target_mod].token_emb(out) + y_emb[:, :cur_len]
            # Build causal mask
            causal_mask = torch.ones((cur_len, cur_len), dtype=torch.bool, device=y.device).triu(1)
            causal_mask = repeat(causal_mask, "n1 n2 -> b n1 n2", b=B)
            
            ### 3a - Decoder forward pass, with conditioning
            y_cond = self.model.forward_decoder(y, context_cond, encoder_mask_cond, causal_mask)
            logits_cond = self.model.forward_logits(y_cond, decoder_mod_dict_cond, decoder_mod_mask_cond[:, :cur_len])[target_mod]
            logits_cond = rearrange(logits_cond, "(b n) d -> b n d", b=B, n=cur_len)
            last_logits_cond = logits_cond[:, -1]

            ### 3b - Decoder forward pass, without conditioning
            y_uncond = self.model.forward_decoder(y, context_uncond, encoder_mask_uncond, causal_mask)
            logits_uncond = self.model.forward_logits(y_uncond, decoder_mod_dict_uncond, decoder_mod_mask_uncond[:, :cur_len])[target_mod]
            logits_uncond = rearrange(logits_uncond, "(b n) d -> b n d", b=B, n=cur_len)
            last_logits_uncond = logits_uncond[:, -1]

            ### 3c - Classifier-free guidance
            last_logits = last_logits_uncond + (last_logits_cond - last_logits_uncond) * guidance_scale

            # Sample token for the newly generated logit
            if np.isclose(temperature, 0, atol=1e-10):
                sample = torch.argmax(last_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = self.top_k_top_p_filtering(last_logits, top_k, top_p)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim=-1)

            if use_eos and (out == eos_token).any(dim=-1).all():
                break

        mod_dict = self.merge_sequences_batched(mod_dict, out, target_mod, text_tokenizer)

        return mod_dict


    @torch.no_grad()
    def generate(self, mod_dict, schedule, top_k=0.0, top_p=0.0, text_tokenizer=None, verbose=False, seed=None):
        """ Generates a sequence of tokens from the input modalities.
        :param mod_dict: Dictionary of modalities.
        :param schedule: Schedule of modalities to use. 
            List of dictionaries containing {target_domain, scheme, num_tokens, temperature, cfg_scale, cfg_cond_domains}.
        :param top_k: top_k > 0: Keep only top k tokens with highest probability (a.k.a. top-k filtering).
        :param top_p: top_p > 0.0: Keep the top tokens with cumulative probability >= top_p (a.k.a. nucleus filtering).
        :param text_tokenizer: Text tokenizer.
        :param verbose: Whether to print progress.
        :param seed: Random seed.
        :return: Generated mod dict.
        """

        # Input embedding -> tokenizes the modalities - Many are placeholder for now
        mod_dict = copy.deepcopy(mod_dict)

        for step, schedule_step_info in tqdm(enumerate(schedule), disable=not verbose):
            target_mod = schedule_step_info['target_domain']
            temp = schedule_step_info['temperature']
            cfg_scale = schedule_step_info.get('cfg_scale', 1.0)
            cfg_conditioning = schedule_step_info.get('cfg_cond_domains', [])
            seed_i = seed + step if seed is not None else None
            
            if self.model.modality_info[target_mod]['type'] == 'img':
                scheme = schedule_step_info['scheme']
                num_select = schedule_step_info['num_tokens']    

                if scheme.lower() == 'maskgit':
                    if cfg_scale == 1.0 or len(cfg_conditioning) == 0:
                        mod_dict = self.maskgit_step_batched(
                            mod_dict, target_mod, num_select, temperature=temp, 
                            top_k=top_k, top_p=top_p, seed=seed_i
                        )
                    else:
                        mod_dict = self.guided_maskgit_step_batched(
                            mod_dict, target_mod, num_select, temperature=temp, top_k=top_k, top_p=top_p,
                            conditioning=cfg_conditioning, guidance_scale=cfg_scale, seed=seed_i
                        )
                elif scheme.lower() == 'roar':
                    if cfg_scale == 1.0 or len(cfg_conditioning) == 0:
                        mod_dict = self.roar_step_batched(
                            mod_dict, target_mod, num_select, temperature=temp, 
                            top_k=top_k, top_p=top_p, seed=seed_i
                        )
                    else:
                        mod_dict = self.guided_roar_step_batched(
                            mod_dict, target_mod, num_select, temperature=temp, top_k=top_k, top_p=top_p, 
                            conditioning=cfg_conditioning, guidance_scale=cfg_scale, seed=seed_i
                        )
                else:
                    raise ValueError("Invalid sampling scheme")
            elif self.model.modality_info[target_mod]['type'] in ['seq', 'seq_token']:
                if cfg_scale == 1.0 or len(cfg_conditioning) == 0:
                    mod_dict = self.autoregressive_step_batched(
                        mod_dict, target_mod, temperature=temp, top_k=top_k, top_p=top_p,
                        text_tokenizer=text_tokenizer, seed=seed_i
                    )
                else:
                    mod_dict = self.guided_autoregressive_step_batched(
                        mod_dict, target_mod, temperature=temp, top_k=top_k, top_p=top_p,
                        text_tokenizer=text_tokenizer, conditioning=cfg_conditioning, 
                        guidance_scale=cfg_scale, seed=seed_i
                    )
            else:
                raise ValueError("Invalid schedule")

        return mod_dict


    @torch.no_grad()
    def generate_iter(self, mod_dict, schedule, top_k=0.0, top_p=0.0, text_tokenizer=None, verbose=False, seed=None):
        """ Iterator that generates a sequence of tokens from the input modalities step by step.
        :param mod_dict: Dictionary of modalities.
        :param schedule: Schedule of modalities to use. 
            List of dictionaries containing {target_domain, scheme, num_tokens, temperature, cfg_scale, cfg_cond_domains}.
        :param top_k: top_k > 0: Keep only top k tokens with highest probability (a.k.a. top-k filtering).
        :param top_p: top_p > 0.0: Keep the top tokens with cumulative probability >= top_p (a.k.a. nucleus filtering).
        :param text_tokenizer: Text tokenizer.
        :param verbose: Whether to print progress.
        :param seed: Random seed.
        :return: Iterator of generated mod dict.
        """

        # Input embedding -> tokenizes the modalities - Many are placeholder for now
        mod_dict = copy.deepcopy(mod_dict)

        for step, schedule_step_info in tqdm(enumerate(schedule), disable=not verbose):
            target_mod = schedule_step_info['target_domain']
            temp = schedule_step_info['temperature']
            cfg_scale = schedule_step_info.get('cfg_scale', 1.0)
            cfg_conditioning = schedule_step_info.get('cfg_cond_domains', [])
            seed_i = seed + step if seed is not None else None
            
            if self.model.modality_info[target_mod]['type'] == 'img':
                scheme = schedule_step_info['scheme']
                num_select = schedule_step_info['num_tokens']    

                if scheme.lower() == 'maskgit':
                    if cfg_scale == 1.0 or len(cfg_conditioning) == 0:
                        mod_dict = self.maskgit_step_batched(
                            mod_dict, target_mod, num_select, temperature=temp, 
                            top_k=top_k, top_p=top_p, seed=seed_i
                        )
                    else:
                        mod_dict = self.guided_maskgit_step_batched(
                            mod_dict, target_mod, num_select, temperature=temp, top_k=top_k, top_p=top_p,
                            conditioning=cfg_conditioning, guidance_scale=cfg_scale, seed=seed_i,
                            write_all_predictions=True
                        )
                elif scheme.lower() == 'roar':
                    if cfg_scale == 1.0 or len(cfg_conditioning) == 0:
                        mod_dict = self.roar_step_batched(
                            mod_dict, target_mod, num_select, temperature=temp, 
                            top_k=top_k, top_p=top_p, seed=seed_i
                        )
                    else:
                        mod_dict = self.guided_roar_step_batched(
                            mod_dict, target_mod, num_select, temperature=temp, top_k=top_k, top_p=top_p, 
                            conditioning=cfg_conditioning, guidance_scale=cfg_scale, seed=seed_i
                        )
                else:
                    raise ValueError("Invalid sampling scheme")
            elif self.model.modality_info[target_mod]['type'] in ['seq', 'seq_token']:
                if cfg_scale == 1.0 or len(cfg_conditioning) == 0:
                    mod_dict = self.autoregressive_step_batched(
                        mod_dict, target_mod, temperature=temp, top_k=top_k, top_p=top_p,
                        text_tokenizer=text_tokenizer, seed=seed_i
                    )
                else:
                    mod_dict = self.guided_autoregressive_step_batched(
                        mod_dict, target_mod, temperature=temp, top_k=top_k, top_p=top_p,
                        text_tokenizer=text_tokenizer, conditioning=cfg_conditioning, 
                        guidance_scale=cfg_scale, seed=seed_i
                    )
            else:
                raise ValueError("Invalid schedule")

            yield mod_dict

    @torch.no_grad()
    def generate_multi_guided(self, uncond_dict, cond_dicts, schedule, top_k=0.0, top_p=0.0, 
                              text_tokenizer=None, verbose=False, seed=None):
        # Generation function for multiple weighted conditions

        # To detect when a modality has finished generating, we keep track of the current target modality
        cur_target_mod = schedule[0]['target_domain']

        uncond_dict = copy.deepcopy(uncond_dict)
        cond_dicts = copy.deepcopy(cond_dicts)

        # Add the to-be-generated modality to the conditional dicts
        for i in range(len(cond_dicts)):
            cond_dicts[i][cur_target_mod] = copy.deepcopy(uncond_dict[cur_target_mod])

        for step, schedule_step_info in tqdm(enumerate(schedule), disable=not verbose):
            target_mod = schedule_step_info['target_domain']
            temp = schedule_step_info['temperature']
            num_select = schedule_step_info['num_tokens']
            cond_weights = schedule_step_info['cfg_scale']

            # Once a modality is fully generated, add it as a new condition
            if cur_target_mod != target_mod:
                for i in range(len(cond_dicts)):
                    # Remove the previously generated modality from the conditionings
                    del cond_dicts[i][cur_target_mod]
                    # Add the next modality to be generated to the conditionings
                    cond_dicts[i][target_mod] = copy.deepcopy(uncond_dict[target_mod])
                
                # Remove the fully generated modality from the unconditional dict inputs
                uncond_dict[cur_target_mod]['input_mask'][:] = True

                # Add the previously generated modality as an additional condition
                new_cond = {}
                new_cond[cur_target_mod] = copy.deepcopy(uncond_dict[cur_target_mod])
                new_cond[cur_target_mod]['input_mask'][:] = False
                new_cond[cur_target_mod]['target_mask'][:] = True
                new_cond[target_mod] = copy.deepcopy(uncond_dict[target_mod])
                cond_dicts.append(new_cond)

                cur_target_mod = target_mod

            if self.model.modality_info[target_mod]['type'] == 'img':
                scheme = schedule_step_info['scheme']

                if scheme.lower() == 'maskgit':
                    uncond_dict, cond_dicts = self.multi_guided_maskgit_step_batched(
                        uncond_dict, cond_dicts, cond_weights, target_mod, num_select, temp, top_k, top_p, seed=seed
                    )
                elif scheme.lower() == 'roar':
                    uncond_dict, cond_dicts = self.multi_guided_roar_step_batched(
                        uncond_dict, cond_dicts, cond_weights, target_mod, num_select, temp, top_k, top_p, seed=seed
                    )
                else:
                    raise ValueError("Invalid sampling scheme")

            else:
                raise NotImplementedError("Only image modalities are supported for now")

        return uncond_dict