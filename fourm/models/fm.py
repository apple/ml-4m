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
import random
import copy
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch
from einops import rearrange, repeat
from torch import nn
import torch.nn.functional as F

from fourm.utils.timm.registry import register_model
from huggingface_hub import PyTorchModelHubMixin

from .fm_utils import Block, DecoderBlock, LayerNorm
from fourm.data.modality_info import MODALITY_INFO


# Model definitions
__all__ = [
    # GELU models
    'fm_tiny_6e_6d_gelu',
    'fm_small_8e_8d_gelu',
    'fm_base_12e_12d_gelu',
    'fm_large_24e_24d_gelu',
    'fm_xlarge_24e_24d_gelu',
    # SwiGLU models
    'fm_tiny_6e_6d_swiglu_nobias',
    'fm_small_8e_8d_swiglu_nobias',
    'fm_base_12e_12d_swiglu_nobias',
    'fm_large_24e_24d_swiglu_nobias',
    'fm_xlarge_24e_24d_swiglu_nobias',
    # SwiGLU + QKNorm models
    'fm_base_12e_12d_swiglu_qknorm_nobias',
    'fm_large_24e_24d_swiglu_qknorm_nobias',
    'fm_xlarge_24e_24d_swiglu_qknorm_nobias',
]



class FourM(nn.Module):
    """4M model.

    Args:
        encoder_embeddings: Dict of encoder embedding modules.
        decoder_embeddings: Dict of decoder embedding modules.
        modality_info: Dict containing modality information.
        dim: Embedding dimension.
        encoder_depth: Number of encoder blocks.
        decoder_depth: Number of decoder blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: If True, add a learnable bias to query, key, value projections.
        proj_bias: If True, add a learnable bias to the last projection of the attention block.
        mlp_bias: If True, add a learnable bias to linear layers in the MLP / feed-forward.
        drop_path_rate_encoder: Stochastic depth rate for encoder.
        drop_path_rate_decoder: Stochastic depth rate for decoder.
        shared_drop_path: If True, shares drop path between encoder and decoder.
        act_layer: Activation layer to be used.
        norm_layer: Normalization layer to be used.
        gated_mlp: If True, make the feedforward gated (e.g., SwiGLU).
        qk_norm: If True, applies normalization to queries and keys (QKNorm).
        decoder_causal_mask: If True, decoder will use a causal mask for all tokens.
        decoder_sep_mask: If True, decoder attention is restricted to within each modality only.
        num_register_tokens: Number of register tokens.
        use_act_checkpoint: If True, use activation checkpoint for each block.
    """
    def __init__(self,
                 encoder_embeddings: Dict[str, nn.Module],
                 decoder_embeddings: Dict[str, nn.Module],
                 modality_info: Dict[str, Any],
                 dim: int = 768,
                 encoder_depth: int = 12,
                 decoder_depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 proj_bias: bool = True,
                 mlp_bias: bool = True,
                 drop_path_rate_encoder: float = 0.0,
                 drop_path_rate_decoder: float = 0.0,
                 shared_drop_path: bool = False,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: Union[partial, nn.Module] = partial(LayerNorm, eps=1e-6),
                 gated_mlp: bool = False, # Make the feedforward gated for e.g. SwiGLU
                 qk_norm: bool = False,
                 decoder_causal_mask: bool = False,
                 decoder_sep_mask: bool = True,
                 num_register_tokens: int = 0,
                 use_act_checkpoint: bool = False,
                 share_modality_embeddings: bool = True,
                 ):
        super().__init__()

        self.modality_info = modality_info
        self.dim = dim
        self.decoder_causal_mask = decoder_causal_mask
        self.decoder_sep_mask = decoder_sep_mask
        self.init_std = 0.02
        self.use_act_checkpoint = use_act_checkpoint
        self.num_register_tokens = num_register_tokens


        # Encoder embeddings & init
        self.encoder_modalities = set(encoder_embeddings.keys())
        for emb in encoder_embeddings.values():
            emb.init(dim_tokens=dim, init_std=self.init_std)
        self.encoder_embeddings = nn.ModuleDict(encoder_embeddings)

        # Decoder embeddings & init
        self.decoder_modalities = set(decoder_embeddings.keys())
        for emb in decoder_embeddings.values():
            emb.init(dim_tokens=dim, init_std=self.init_std)
        self.decoder_embeddings = nn.ModuleDict(decoder_embeddings)

        # Share modality embeddings across the encoder and decoder embedding modules
        if share_modality_embeddings:
            self.share_modality_embeddings()

        ## Transformer encoder
        if shared_drop_path:
            dpr_encoder = [x.item() for x in torch.linspace(0, drop_path_rate_encoder, encoder_depth + decoder_depth)][:encoder_depth]
        else:
            dpr_encoder = [x.item() for x in torch.linspace(0, drop_path_rate_encoder, encoder_depth)] # stochastic depth decay rule

        self.encoder = nn.ModuleList([
            Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias, mlp_bias=mlp_bias,
                 drop_path=dpr_encoder[i], act_layer=act_layer, norm_layer=norm_layer, gated_mlp=gated_mlp, qk_norm=qk_norm)
            for i in range(encoder_depth)
        ])
        self.encoder_norm = norm_layer(dim)


        ## Transformer decoder
        if shared_drop_path:
            dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate_decoder, encoder_depth + decoder_depth)][encoder_depth:]
        else:
            dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate_decoder, decoder_depth)]  # stochastic depth decay rule

        # Projection of encoder tokens before adding the embeddings again
        self.decoder_proj_context = nn.Linear(dim, dim)

        self.decoder = nn.ModuleList([
            DecoderBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias, mlp_bias=mlp_bias, 
                         drop_path=dpr_decoder[i], act_layer=act_layer, norm_layer=norm_layer, gated_mlp=gated_mlp, qk_norm=qk_norm)
            for i in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=self.init_std)

        # Additional register tokens that can be used by the encoder during fine-tuning
        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, dim))
            nn.init.normal_(self.register_tokens, std=self.init_std)
        else:
            self.register_tokens = None

        # Weight init
        self.init_weights()

    def share_modality_embeddings(self):
        """Share modality embeddings across the encoder and decoder embedding modules."""
        shared_modalities = self.encoder_modalities & self.decoder_modalities
        for mod in shared_modalities:
            self.decoder_embeddings[mod].mod_emb = self.encoder_embeddings[mod].mod_emb

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
                if m.bias is not None:
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

    def get_num_layers_decoder(self):
        return len(self.decoder)

    def get_num_layers(self):
        return self.get_num_layers_encoder() + self.get_num_layers_decoder()

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = set()

        for mod, emb_module in self.encoder_embeddings.items():
            if hasattr(emb_module, 'no_weight_decay'):
                to_skip = emb_module.no_weight_decay()
                to_skip = set([f'encoder_embeddings.{mod}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        for mod, emb_module in self.decoder_embeddings.items():
            if hasattr(emb_module, 'no_weight_decay'):
                to_skip = emb_module.no_weight_decay()
                to_skip = set([f'decoder_embeddings.{mod}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        return no_wd_set

    def cat_encoder_tensors(self, mod_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        """Concatenate encoder tensors from different modalities.

        Args:
            mod_dict (dict): A dictionary containing information for each modality. 
                             Expected keys for each modality are 'x' (input tokens), 
                             'emb' (embeddings), 'input_mask', etc.

        Returns:
            tuple:
                - encoder_tokens_all (torch.Tensor): Concatenated encoder tokens from all modalities. Shape (B, O, D) where O is the total number of all encoder tokens.
                - emb_all (torch.Tensor): Concatenated encoder embeddings from all modalities. Shape (B, O, D)
                - encoder_mask_all (torch.Tensor): Concatenated boolean masks indicating which tokens are part of the encoder input (set to 0 for valid tokens, 1 otherwise). Shape (B, O)
                - mod_mask_all (torch.Tensor): Concatenated integer mask marking the modality type for each encoder token. Shape (B, O)
        """

        encoder_tokens_all = []
        emb_all = []
        encoder_mask_all = []
        mod_mask_all = []

        for mod, d in mod_dict.items():
            encoder_tokens_all.append(d['x'])
            emb_all.append(d['emb'])
            encoder_mask_all.append(d['input_mask'])
            mod_mask_all.append(torch.full_like(d['input_mask'], self.modality_info[mod]['id'], dtype=torch.int16))

        encoder_tokens_all = torch.cat(encoder_tokens_all, dim=1)
        emb_all = torch.cat(emb_all, dim=1)
        encoder_mask_all = torch.cat(encoder_mask_all, dim=1)
        mod_mask_all = torch.cat(mod_mask_all, dim=1)

        return encoder_tokens_all, emb_all, encoder_mask_all, mod_mask_all

    def cat_decoder_tensors(self, mod_dict: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor]:
        """Concatenate decoder tensors from different modalities.
        
        Args:
            mod_dict (dict): A dictionary containing information for each modality.
                             Expected keys for each modality include 'x' (input tokens),
                             'ids' (target IDs), 'emb' (embeddings), 'target_mask', 'decoder_attention_mask', etc.

        
        Returns:
            tuple:
                - decoder_tokens_all (torch.Tensor): Concatenated decoder tokens from all modalities. Shape (B, P, D) where P is the total number of all decoder tokens.
                - emb_all (torch.Tensor): Concatenated decoder embeddings from all modalities. Shape (B, P, D)
                - decoder_mask_all (torch.Tensor): Concatenated boolean masks indicating which tokens are part of the decoder input / target (set to 0 for valid tokens, 1 otherwise). Shape (B, P)
                - target_ids_all (torch.Tensor): Concatenated target IDs from all modalities. Shape (B, P)
                - attention_mask_all (torch.Tensor): Concatenated attention masks in compressed format, needs to be passed to adapt_decoder_attention_mask() to obtain the final attention mask. Shape (B, P)
                - mod_mask_all (torch.Tensor): Concatenated integer mask marking the modality type for each decoder token. Shape (B, P)
        """

        decoder_tokens_all = []
        target_ids_all = []
        emb_all = []
        decoder_mask_all = []
        attention_mask_all = []
        mod_mask_all = []

        # Shuffle order in which modalities are provided (useful for modality causal mask)
        mod_dict = {mod: d for mod, d in random.sample(mod_dict.items(), len(mod_dict))}

        for mod, d in mod_dict.items():
            if self.modality_info[mod]['type'] in ['seq', 'seq_emb', 'seq_token']:
                # Important: This makes the assumption that the target sequence appears sequentially
                # before sorting / gathering
                decoder_tokens_all.append(d['x'][:, :-1])
                target_ids_all.append(d['ids'][:, 1:])  # Shifted left
                emb_all.append(d['emb'][:, :-1])
                # Logical or with left shifting removes the last unmasked position
                decoder_mask_all.append(torch.logical_or(d['target_mask'][:, 1:], d['target_mask'][:, :-1]))
                # Add attention mask ids
                attention_mask_all.append(d['decoder_attention_mask'][:, :-1])
                mod_mask_all.append(torch.full_like(d['ids'][:, :-1], self.modality_info[mod]['id'], dtype=torch.int16))
            else:
                # Important: For 2d / image modalities, the decoder input tokens are replaced by the mask token
                decoder_tokens_all.append(torch.zeros_like(d['x']) + self.mask_token)  # Replace x by mask token
                target_ids_all.append(d['ids'])
                emb_all.append(d['emb'])
                decoder_mask_all.append(d['target_mask'])
                attention_mask_all.append(d['decoder_attention_mask'])
                mod_mask_all.append(torch.full_like(d['ids'], self.modality_info[mod]['id'], dtype=torch.int16))

        decoder_tokens_all = torch.cat(decoder_tokens_all, dim=1)
        emb_all = torch.cat(emb_all, dim=1)
        decoder_mask_all = torch.cat(decoder_mask_all, dim=1)
        target_ids_all = torch.cat(target_ids_all, dim=1)
        attention_mask_all = torch.cat(attention_mask_all, dim=1)
        mod_mask_all = torch.cat(mod_mask_all, dim=1)

        return decoder_tokens_all, emb_all, decoder_mask_all, target_ids_all, attention_mask_all, mod_mask_all

    def forward_mask_encoder(self, mod_dict: Dict[str, Dict[str, torch.Tensor]], num_encoder_tokens: int) -> Tuple[torch.Tensor]:
        """Concatenates and mask encoder tensors based on provided modality information.

        This function consolidates encoder tokens from multiple modalities, then selects a specified number of them based on modality information (i.e. masking).

        Args:
            mod_dict (dict): Dictionary containing tensors for different modalities. 
                            It is expected to have keys for each modality and values 
                            containing the modalities' associated tensors.
            num_encoder_tokens (int): Number of encoder tokens to retain after masking.

        Returns:
            tuple:
                - encoder_tokens (torch.Tensor): Selected encoder tokens from all modalities. Shape (B, N, D) where N is the number of selected encoder tokens. 
                - encoder_emb (torch.Tensor): Corresponding embeddings for encoder tokens. Shape (B, N, D)
                - encoder_mask (torch.Tensor): A boolean mask indicating which encoder tokens are valid (set to 0 for valid tokens, 1 otherwise). Shape (B, 1, N)
                - mod_mask (torch.Tensor): An integer mask marking the modality type for each encoder token (with -1 indicating unassigned pad tokens). Shape (B, N)

        Notes:
            - If `num_register_tokens` is set and greater than 0, register tokens are added at the beginning of the sequence.
        """
        B = list(mod_dict.values())[0]['tensor'].shape[0]

        encoder_tokens_all, emb_all, encoder_mask_all, mod_mask_all = self.cat_encoder_tensors(mod_dict)

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

        if self.num_register_tokens > 0:
            register_tokens = repeat(self.register_tokens, '() n d -> b n d', b=B)
            # We add register tokens at the beginning of the sequence
            encoder_tokens = torch.cat([register_tokens, encoder_tokens], dim=1)
            encoder_emb = torch.cat([torch.zeros_like(register_tokens), encoder_emb], dim=1)
            encoder_mask = torch.cat([torch.zeros((B, register_tokens.shape[1]), dtype=torch.bool, device=encoder_mask.device), encoder_mask], dim=1)
            mod_mask = torch.cat([torch.full((B, register_tokens.shape[1]), -1, dtype=torch.int16, device=mod_mask.device), mod_mask], dim=1)

        encoder_tokens[encoder_mask] = 0.
        encoder_emb[encoder_mask] = 0.
        mod_mask[encoder_mask] = -1
        # Mask could be of shape 'b n1 n2' but not needed for masked_fill
        # This means this mask can then be re-used for decoder cross-attention
        encoder_mask = rearrange(encoder_mask, 'b n2 -> b 1 n2')

        return encoder_tokens, encoder_emb, encoder_mask, mod_mask

    def forward_mask_decoder(self, mod_dict: Dict[str, Dict[str, torch.Tensor]], num_decoder_tokens: int) -> Tuple[torch.Tensor]:
        """Concatenates and mask decoder tensors based on provided modality information.

        This function consolidates decoder tokens from multiple modalities, selects a specified number of them based on modality information, and applies appropriate masking.

        Args:
            mod_dict (dict): Dictionary containing tensors for different modalities.
                            It is expected to have keys for each modality and values 
                            containing the modalities' associated tensors.
            num_decoder_tokens (int): Number of decoder tokens to retain after masking.

        Returns:
            tuple:
                - decoder_tokens (torch.Tensor): Selected decoder tokens from all modalities. Shape (B, M, D) where M is the number of selected decoder tokens.
                - decoder_emb (torch.Tensor): Corresponding embeddings for decoder tokens. Shape (B, M, D)
                - decoder_mask (torch.Tensor): A boolean mask indicating which decoder tokens are valid (set to 0 for valid tokens, 1 otherwise). Shape (B, 1, M)
                - target_ids (torch.Tensor): IDs of the target tokens corresponding to the decoder tokens. Shape (B, M)
                - decoder_attention_mask (torch.Tensor): Mask for the decoder self-attention layers. Shape (B, M, M)
                - mod_mask (torch.Tensor): An integer mask marking the modality type for each decoder token (with -1 indicating unassigned pad tokens). Shape (B, M)
        """
        # decoder_mask and target_mask are equivalent, we rename it here to harmonize with forward_mask_encoder
        decoder_tokens_all, emb_all, decoder_mask_all, target_ids_all, decoder_attention_mask_all, mod_mask_all = self.cat_decoder_tensors(mod_dict)

        # Add arange multiplied by small constant to mask so they get sorted in a deterministic way
        mask_arange = torch.arange(decoder_mask_all.shape[1], device=decoder_mask_all.device).unsqueeze(0) * 1e-6
        ids_shuffle = torch.argsort(decoder_mask_all + mask_arange, dim=1)
        # ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_decoder_tokens]

        decoder_tokens = torch.gather(decoder_tokens_all, dim=1, index=repeat(ids_keep, "b n -> b n d", d=decoder_tokens_all.shape[2]))
        decoder_emb = torch.gather(emb_all, dim=1, index=repeat(ids_keep, "b n -> b n d", d=emb_all.shape[2]))
        decoder_mask = torch.gather(decoder_mask_all, dim=1, index=ids_keep)
        target_ids = torch.gather(target_ids_all, dim=1, index=ids_keep)
        decoder_attention_mask = torch.gather(decoder_attention_mask_all, dim=1, index=ids_keep)
        mod_mask = torch.gather(mod_mask_all, dim=1, index=ids_keep)

        decoder_tokens[decoder_mask] = 0.
        decoder_emb[decoder_mask] = 0.
        target_ids[decoder_mask] = 0
        decoder_attention_mask = self.adapt_decoder_attention_mask(decoder_attention_mask, mod_mask)
        mod_mask[decoder_mask] = -1

        # This means this mask can then be re-used for decoder cross-attention
        decoder_mask = rearrange(decoder_mask, 'b n2 -> b 1 n2')


        return decoder_tokens, decoder_emb, decoder_mask, target_ids, decoder_attention_mask, mod_mask

    def adapt_decoder_attention_mask(self, decoder_attention_mask: torch.Tensor, mod_mask=Optional[torch.Tensor]) -> torch.Tensor:
        """
        Transforms the compressed decoder attention mask to a full attention mask based on the specified constraints.

        Args:
            decoder_attention_mask (torch.Tensor): Initial attention mask indicating attention constraints. Shape (B, M) where M is the number of the decoder tokens.
            mod_mask (torch.Tensor, optional): Modality mask to separate attention masks per modality. Shape (B, M)

        Returns:
            torch.Tensor: Adapted attention mask. Shape (B, M, M) where M is the number of the decoder tokens.
        """
        B, N = decoder_attention_mask.shape

        if self.decoder_causal_mask:
            # For causal mode, tokens can only attend to preceding tokens and themselves.
            causal_mask = torch.ones((N, N), dtype=torch.bool, device=decoder_attention_mask.device).triu(1)
            causal_mask = repeat(causal_mask, "n1 n2 -> b n1 n2", b=B)
            adapted_attention_mask = causal_mask
        else:
            # Cumulatively sum the attention mask to determine token-wise attention behavior.
            # Examples:
            # Mask [4, 0, 0, 0] -> Cumsum: [4, 4, 4, 4] -> All tokens attend to each other.
            # Mask [1, 1, 1, 1] -> Cumsum: [1, 2, 3, 4] -> Strict autoregressive behavior.
            # Mask [2, 0, 1, 1] -> Cumsum: [2, 2, 3, 4] -> Tokens 1 and 2 attend to each other, token 3 attends to tokens 1-3, and token 4 to all.
            attention_arange = torch.arange(N, device=decoder_attention_mask.device)
            attention_arange = repeat(attention_arange, "n2 -> b n1 n2", b=B, n1=N)
            cumsum_mask = torch.cumsum(decoder_attention_mask, dim=-1)
            cumsum_mask = rearrange(cumsum_mask, "b n -> b n 1")
            adapted_attention_mask = (attention_arange >= cumsum_mask)

        if self.decoder_sep_mask:
            # Separate attention between tokens based on their modality using mod_mask.
            sep_mask = repeat(mod_mask, "b n2 -> b n1 n2", n1=N) != repeat(mod_mask, "b n1 -> b n1 n2", n2=N)
            adapted_attention_mask = adapted_attention_mask | sep_mask

        return adapted_attention_mask

    def forward_encoder(self, 
                        x: torch.Tensor, 
                        encoder_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for the encoder.
        
        Args:
            x (torch.Tensor): Encoder input tokens. Shape (B, N, D) where N is the number of encoder tokens.
            encoder_mask (torch.Tensor): Encoder mask indicating which tokens are valid (set to 0 for valid tokens, 1 otherwise). Shape (B, 1, N)
            
        Returns:
            torch.Tensor: Encoder output. Shape (B, N, D)
        """

        for blk in self.encoder:
            x = blk(x, mask=encoder_mask)
            
        x = self.encoder_norm(x)

        return x

    def forward_decoder(self, 
                        y: torch.Tensor, 
                        context: torch.Tensor, 
                        encoder_mask: torch.Tensor, 
                        decoder_attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for the decoder.

        Args:
            y (torch.Tensor): Decoder input tokens. Shape (B, M, D).
            context (torch.Tensor): Context for the decoder (i.e. encoder output). Shape (B, N, D).
            encoder_mask (torch.Tensor): Encoder mask indicating which tokens are valid (set to 0 for valid tokens, 1 otherwise). Shape (B, 1, N).
            decoder_attention_mask (torch.Tensor): Decoder attention mask. Shape (B, M, M).

        Returns:
            torch.Tensor: Decoder output. Shape (B, M, D).
        """

        for blk in self.decoder:
            y = blk(y, context, sa_mask=decoder_attention_mask, xa_mask=encoder_mask)

        y = self.decoder_norm(y)

        return y

    def forward_logits(self, 
                       y: torch.Tensor, 
                       decoder_mod_dict: Dict[str, Dict[str, torch.Tensor]], 
                       decoder_mod_mask: torch.Tensor,
                       return_all_logits: bool = False) -> Dict[str, torch.Tensor]:
        """Forward computation of logits for each modality.

        Args:
            y (torch.Tensor): Decoder output. Shape (B, M, D).
            decoder_mod_dict (dict): Dictionary containing tensor information for each modality in the decoder.
            decoder_mod_mask (torch.Tensor): Integer mask indicating which tokens belong to which modality. Shape (B, M).

        Returns:
            Dict[str, torch.Tensor]: Dictionary of logits for each modality.
        """

        mod_logits = {}
        for mod, d in decoder_mod_dict.items():
            idx = self.modality_info[mod]["id"]
            if return_all_logits:
                logits = self.decoder_embeddings[mod].forward_logits(y)
            else:
                logits = self.decoder_embeddings[mod].forward_logits(y[decoder_mod_mask == idx])
            mod_logits[mod] = logits
        return mod_logits

    def forward_loss(self, 
                     y: torch.Tensor, 
                     target_ids: torch.Tensor, 
                     decoder_mod_dict: Dict[str, Any], 
                     decoder_mod_mask: torch.Tensor, loss_type: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes the loss based on the specified loss type.

        Args:
            y (torch.Tensor): Decoder output. Shape (B, M, D).
            target_ids (torch.Tensor): Ground truth token IDs. Shape (B, M).
            decoder_mod_dict (dict): Dictionary containing tensor information for each modality in the decoder.
            decoder_mod_mask (torch.Tensor): Integer mask indicating which tokens belong to which modality. Shape (B, M).
            loss_type (str): The type of loss to compute. Either 'mod' or 'token'.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total loss and dictionary of loss for each modality.
        """
        if loss_type in ['mod', 'modality']:
            loss, mod_loss = self.forward_mod_loss(y, target_ids, decoder_mod_dict, decoder_mod_mask)
        elif loss_type == 'token':
            loss, mod_loss = self.forward_token_loss(y, target_ids, decoder_mod_dict, decoder_mod_mask)
        else:
            raise ValueError("Invalid loss type")

        return loss, mod_loss

    def forward_mod_loss(self, 
                         y: torch.Tensor, 
                         target_ids: torch.Tensor, 
                         decoder_mod_dict: Dict[str, Any], 
                         decoder_mod_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes the modality-wise loss.

        Args:
            y (torch.Tensor): Decoder tokens. Shape (B, M, D).
            target_ids (torch.Tensor): Ground truth token IDs. Shape (B, M).
            decoder_mod_dict (dict): Dictionary containing tensor information for each modality in the decoder.
            decoder_mod_mask (torch.Tensor): Mask indicating which tokens belong to which modality. Shape (B, M).

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total modality loss and dictionary of loss for each modality.
        """       
        mod_loss = {}
        for mod, d in decoder_mod_dict.items():
            idx = self.modality_info[mod]["id"]
            logits = self.decoder_embeddings[mod].forward_logits(y[decoder_mod_mask == idx])
            if logits.numel() == 0:
                # If there are no logits / targets, set mod_loss to 0
                mod_loss[mod] = torch.zeros(1, device=logits.device)
            else:
                loss = F.cross_entropy(logits, target_ids[decoder_mod_mask == idx].long(), reduction='mean')
                mod_loss[mod] = loss

        loss = sum(mod_loss.values()) / len(mod_loss)

        return loss, mod_loss

    def forward_token_loss(self, 
                           y: torch.Tensor, 
                           target_ids: torch.Tensor, 
                           decoder_mod_dict: Dict[str, Any], 
                           decoder_mod_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes the token-wise loss.

        Args:
            y (torch.Tensor): Decoder tokens. Shape (B, M, D).
            target_ids (torch.Tensor): Ground truth token IDs. Shape (B, M).
            decoder_mod_dict (dict): Dictionary containing tensor information for each modality in the decoder.
            decoder_mod_mask (torch.Tensor): Mask indicating which tokens belong to which modality. Shape (B, M).

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total token loss and dictionary of loss for each modality.
        """        
        mod_loss = {}
        mod_count = {}

        for mod, d in decoder_mod_dict.items():
            idx = self.modality_info[mod]["id"]
            logits = self.decoder_embeddings[mod].forward_logits(y[decoder_mod_mask == idx])
            if logits.numel() == 0:
                # If there are no logits / targets, set mod_loss to 0
                mod_loss[mod] = torch.zeros(1, device=logits.device)
                mod_count[mod] = 0
            else:
                loss = F.cross_entropy(logits, target_ids[decoder_mod_mask == idx].long(), reduction='mean')
                mod_loss[mod] = loss
                mod_count[mod] = logits.numel()

        loss = sum([mod_loss[mod] * mod_count[mod] for mod in mod_loss.keys()]) / sum(mod_count.values())

        return loss, mod_loss


    def forward(self, 
            mod_dict: Dict[str, Dict[str, torch.Tensor]], 
            num_encoder_tokens: int, 
            num_decoder_tokens: int, 
            loss_type: str = 'mod', 
            return_logits: bool = False) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass for the model.

        Args:
            mod_dict (Dict[str, Dict[str, torch.Tensor]]): Dictionary containing the tensors, masks, and other info for each modality.
                - mod_dict[modality_name]["tensor_name"]: Shape can vary based on tensor_name and modality.
            num_encoder_tokens (int): Number of tokens to keep for the encoder.
            num_decoder_tokens (int): Number of tokens to keep for the decoder.
            loss_type (str, optional): The type of loss to compute. Can be 'mod' (average of loss per modality) or 'token' (average loss per token). Default is 'mod'.
            return_logits (bool, optional): If True, return the logits. Default is False.

        Returns:
            Union[dict, tuple]: 
                - If return_logits is True: Dictionary of logits for each modality.
                - Otherwise: Tuple containing the total loss and dictionary of loss for each modality.
        """

        # Mod dicts
        encoder_mod_dict = {mod: self.encoder_embeddings[mod](d)
                            for mod, d in mod_dict.items()
                            if mod in self.encoder_embeddings}
        encoder_tokens, encoder_emb, encoder_mask, encoder_mod_mask = self.forward_mask_encoder(encoder_mod_dict, num_encoder_tokens)

        decoder_mod_dict = {mod: self.decoder_embeddings[mod].forward_embed(d)
                            for mod, d in mod_dict.items()
                            if mod in self.decoder_embeddings}
        decoder_tokens, decoder_emb, decoder_mask, target_ids, decoder_attention_mask, decoder_mod_mask = self.forward_mask_decoder(decoder_mod_dict, num_decoder_tokens)

        # Encoder
        x = encoder_tokens + encoder_emb
        x = self.forward_encoder(x, encoder_mask=encoder_mask)

        # Decoder
        context = self.decoder_proj_context(x) + encoder_emb
        y = decoder_tokens + decoder_emb
        y = self.forward_decoder(y, context, encoder_mask=encoder_mask, decoder_attention_mask=decoder_attention_mask)

        # Logits
        if return_logits:
            mod_logits = self.forward_logits(y, decoder_mod_dict, decoder_mod_mask, return_all_logits=True)
            return mod_logits

        # Loss
        loss, mod_loss = self.forward_loss(y, target_ids, decoder_mod_dict, decoder_mod_mask, loss_type)

        return loss, mod_loss


    def freeze_encoder(self, freeze_embeddings=True):
        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder_norm.parameters():
            param.requires_grad = False

        if freeze_embeddings:
            for param in self.encoder_embeddings.parameters():
                param.requires_grad = False

    def freeze_encoder_except_specific_embeddings(self, frozen_embedding_domain):
        frozen_embedding_domain = frozen_embedding_domain.split('-')
        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder_norm.parameters():
            param.requires_grad = False

        for name, param in self.encoder_embeddings.named_parameters():
            if name.split('.')[0] in frozen_embedding_domain:
                param.requires_grad = False

    def unfreeze_encoder(self, unfreeze_embeddings=True):
        for param in self.encoder.parameters():
            param.requires_grad = True

        for param in self.encoder_norm.parameters():
            param.requires_grad = True

        if unfreeze_embeddings:
            for param in self.encoder_embeddings.parameters():
                param.requires_grad = True

    def freeze_decoder(self, freeze_embeddings=True):
        for param in self.decoder.parameters():
            param.requires_grad = False

        for param in self.decoder_norm.parameters():
            param.requires_grad = False

        if freeze_embeddings:
            for param in self.decoder_embeddings.parameters():
                param.requires_grad = False

    def freeze_decoder_except_specific_embeddings(self, frozen_embedding_domain):
        frozen_embedding_domain = frozen_embedding_domain.split('-')
        for param in self.decoder.parameters():
            param.requires_grad = False

        for param in self.decoder_norm.parameters():
            param.requires_grad = False

        for name, param in self.decoder_embeddings.named_parameters():
            if name.split('.')[0] in frozen_embedding_domain:
                param.requires_grad = False

    def unfreeze_decoder(self, unfreeze_embeddings=True):
        for param in self.decoder.parameters():
            param.requires_grad = True

        for param in self.decoder_norm.parameters():
            param.requires_grad = True

        if unfreeze_embeddings:
            for param in self.decoder_embeddings.parameters():
                param.requires_grad = True

    def freeze_shared_params(self):
        self.freeze_encoder(freeze_embeddings=False)
        self.freeze_decoder(freeze_embeddings=False)

    def freeze_params_except_specific_embeddings(self, frozen_embedding_domain):
        self.freeze_encoder_except_specific_embeddings(frozen_embedding_domain=frozen_embedding_domain)
        self.freeze_decoder_except_specific_embeddings(frozen_embedding_domain=frozen_embedding_domain)

    def unfreeze_shared_params(self):
        self.unfreeze_encoder(unfreeze_embeddings=False)
        self.unfreeze_decoder(unfreeze_embeddings=False)

    def unfreeze_all(self):
        self.unfreeze_encoder(unfreeze_embeddings=True)
        self.unfreeze_decoder(unfreeze_embeddings=True)


################################################

# Wrapper for easy loading with Huggingface Hub

class FM(FourM, PyTorchModelHubMixin):
    """Wrapper around FourM for easy loading with Huggingface Hub.

    Args:
        config (dict): Dictionary containing the model and modality configuration, 
            used for loading from Huggingface Hub.
    """
    def __init__(self, config: dict):

        config = copy.deepcopy(config)

        all_domains = sorted(list(set(config['domains_in']) | set(config['domains_out'])))
        modality_info = {mod: MODALITY_INFO[mod] for mod in all_domains}

        encoder_embeddings = {}
        for mod in config['domains_in']:
            info = modality_info[mod]
            if info.get("encoder_embedding", None) is not None:
                if info["type"] == "img":
                    image_size, patch_size = info.get('input_size', config['image_size']), info.get('patch_size', config['patch_size'])
                    encoder_embeddings[mod] = info["encoder_embedding"](patch_size=patch_size, image_size=image_size)
                else:
                    encoder_embeddings[mod] = info["encoder_embedding"]()
    
        decoder_embeddings = {}
        for mod in config['domains_out']:
            info = modality_info[mod]
            if info.get("decoder_embedding", None) is not None:
                if info["type"] == "img":
                    image_size, patch_size = info.get('input_size', config['image_size']), info.get('patch_size', config['patch_size'])
                    decoder_embeddings[mod] = info["decoder_embedding"](patch_size=patch_size, image_size=image_size, share_embedding=False)
                else:
                    decoder_embeddings[mod] = info["decoder_embedding"](share_embedding=False)

        config['norm_layer'] = partial(LayerNorm, eps=1e-6, bias=config['norm_bias'])
        config['act_layer'] = getattr(torch.nn, config['act_layer'])

        del config['norm_bias']
        del config['domains_in']
        del config['domains_out']
        del config['image_size']
        del config['patch_size']
        
        super().__init__(
            encoder_embeddings=encoder_embeddings,
            decoder_embeddings=decoder_embeddings,
            modality_info=modality_info,
            **config
        )   


################################################

# Model definitions
        
# GELU variants
@register_model
def fm_tiny_6e_6d_gelu(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=6,
        decoder_depth=6,
        dim=384,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


@register_model
def fm_small_8e_8d_gelu(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=8,
        decoder_depth=8,
        dim=512,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


@register_model
def fm_base_12e_12d_gelu(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=12,
        decoder_depth=12,
        dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


@register_model
def fm_large_24e_24d_gelu(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=24,
        decoder_depth=24,
        dim=1024,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

@register_model
def fm_xlarge_24e_24d_gelu(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=24,
        decoder_depth=24,
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
def fm_tiny_6e_6d_swiglu_nobias(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=6,
        decoder_depth=6,
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
def fm_small_8e_8d_swiglu_nobias(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=8,
        decoder_depth=8,
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
def fm_base_12e_12d_swiglu_nobias(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=12,
        decoder_depth=12,
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
def fm_large_24e_24d_swiglu_nobias(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=24,
        decoder_depth=24,
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
def fm_xlarge_24e_24d_swiglu_nobias(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=24,
        decoder_depth=24,
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
def fm_base_12e_12d_swiglu_qknorm_nobias(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=12,
        decoder_depth=12,
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
def fm_large_24e_24d_swiglu_qknorm_nobias(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=24,
        decoder_depth=24,
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
def fm_xlarge_24e_24d_swiglu_qknorm_nobias(
        encoder_embeddings: Dict[str, nn.Module],
        decoder_embeddings: Dict[str, nn.Module],
        **kwargs):
    model = FourM(
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        encoder_depth=24,
        decoder_depth=24,
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