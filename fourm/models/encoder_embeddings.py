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
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .fm_utils import build_1d_sincos_posemb, build_2d_sincos_posemb, pair

class SequenceEncoderEmbedding(nn.Module):
    """Embedding module for encoding sequence inputs, like captions or a sequence of objects.

    Args:
        vocab_size: Vocabulary size
        max_length: Maximum number of tokens in the sequence
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 1D sin-cos positional embeddings
        max_sincos_pos_emb: Maximum allowed length for sin-cos positional embeddings
        padding_idx: Padding index for word embedding
    """

    def __init__(self,
                 vocab_size: int,
                 max_length: int,
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 max_sincos_pos_emb: int = 512,
                 padding_idx: int = 0,
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.padding_idx = padding_idx
        self.max_sincos_pos_emb = max_sincos_pos_emb

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of embedding module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        if self.sincos_pos_emb:
            if self.max_length > self.max_sincos_pos_emb:
                raise ValueError(f"Max length ({self.max_length}) is greater than the number of posembs ({self.max_sincos_pos_emb}")
            pos_emb = build_1d_sincos_posemb(max_len=self.max_sincos_pos_emb, embed_dim=self.dim_tokens)[:self.max_length]
            self.register_buffer("pos_emb", pos_emb) # self.pos_emb is now a buffer for FSDP
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.max_length, self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding
        self.token_emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.dim_tokens,
                                     padding_idx=self.padding_idx)


    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d : Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming sequence of ids to sequence of embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict with at least the following keys:
                - 'tensor' (torch.Tensor): Input token sequence for each batch. Shape (B, L) where B is the batch size and L is the sequence length.
                - 'input_mask' (torch.Tensor): Mask for valid tokens in the input sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, L, D) where D is the embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, L, D).
        """
        ids = d['tensor']
        B = ids.shape[0]
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'

        # Map to embedding
        x = self.token_emb(ids)

        expanded_pos_emb = repeat(self.pos_emb, "() n d -> b n d", b=B)
        # Input pos encoding
        input_mask = d['input_mask']
        input_pos_id = (~input_mask).int().cumsum(dim=1) - 1
        input_pos_id[input_mask] = 0
        input_pos_emb = torch.gather(expanded_pos_emb, dim=1, index=repeat(input_pos_id, "b n -> b n d", d=expanded_pos_emb.shape[2]))
        input_pos_emb[input_mask] = 0

        x_emb = input_pos_emb + self.mod_emb

        d['x'] = x
        d['emb'] = x_emb
        return d
    
class ImageTokenEncoderEmbedding(nn.Module):
    """Embedding module for tokenized spatial inputs.

    Args:
        vocab_size: Vocabulary size
        patch_size: Int or tuple of the patch size over the full image size.
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
        image_size: Default image size. Used to initialize size of positional embeddings.
    """
    def __init__(self,
                 vocab_size: int,
                 patch_size: Union[int, Tuple[int,int]] = 16,
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 image_size: Union[int, Tuple[int]] = 224,
                 **kwargs):

        super().__init__()
        self.vocab_size = vocab_size
        self.patch_size = pair(patch_size)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // patch_size) * (self.image_size[1] // patch_size)

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // self.patch_size[0]
        w_posemb = self.image_size[1] // self.patch_size[1]
        if self.sincos_pos_emb:
            pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.register_buffer("pos_emb", pos_emb) # self.pos_emb is now a buffer for FSDP
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, (h_posemb * w_posemb), self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding
        self.token_emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.dim_tokens)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming image tokens to a sequence of embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict with at least the following key:
                - 'tensor' (torch.Tensor): Input image tokens for each batch. Shape (B, H, W) where B is the batch size, and H, W are height and width of the tokenized image.                - 'input_mask' (torch.Tensor): Mask for valid tokens in the input sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            Dict[str, torch.Tensor]: Modality dictionary with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, H*W, D).
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, H*W, D).
        """
        ids = d['tensor']
        B = ids.shape[0]
        ids = ids.reshape(B, -1)

        # Map to embedding
        x = self.token_emb(ids)

        # Create positional embedding + modality embedding
        x_emb = repeat(self.pos_emb + self.mod_emb, '() n d -> b n d', b=B)

        d['x'] = x
        d['emb'] = x_emb

        return d


class ImageEncoderEmbedding(nn.Module):
    """Embedding module for spatial inputs, like images or feature maps.
    Creates tokens from patches over the image.

    This adapter / embedding differs from the one of MultiMAE by taking as input a dict and
     separating positional embeddings and modality embeddings from the input projection
     Input projection is 'x', posemb + modemb is 'emb'

    Args:
        num_channels: Number of input channels of the image/feature map
        patch_size: Int or tuple of the patch size over the full image size.
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
        image_size: Default image size. Used to initialize size of positional embeddings.
    """
    def __init__(self,
                 num_channels: int,
                 patch_size: Union[int, Tuple[int,int]],
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 image_size: Union[int, Tuple[int]] = 224):

        super().__init__()
        self.num_channels = num_channels
        self.patch_size = pair(patch_size)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // patch_size) * (self.image_size[1] // patch_size)

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of encoder that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // self.patch_size[0]
        w_posemb = self.image_size[1] // self.patch_size[1]
        if self.sincos_pos_emb:
            pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.register_buffer("pos_emb", pos_emb) # self.pos_emb is now a buffer for FSDP
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, (h_posemb * w_posemb), self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Image -> tokens projection
        # No bias term here, so modality embedding fully comes from self.mod_emb
        self.proj = nn.Linear(self.num_channels * self.patch_size[0] * self.patch_size[1], self.dim_tokens, bias=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming image to sequence of tokens.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict with at least the following key:
                - 'tensor' (torch.Tensor): Input image for each batch. Shape (B, C, H, W) where B is the batch size, C is the number of channels, and H, W are height and width of the image.

                
        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, (H / PH) * (W / PW), D), where PH and PW are the patch sizes
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, (H / PH) * (W / PW), D)
        """
        x = d['tensor']
        B, C, H, W = x.shape
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'
        assert (H % self.patch_size[0] == 0) and (W % self.patch_size[1] == 0), f'Image sizes {H}x{W} must be divisible by patch sizes {self.patch_size[0]}x{self.patch_size[1]}'

        # Create patches [B, C, H, W] -> [B, (H*W), C]
        x_patch = self.proj(rearrange(x, 'b d (nh ph) (nw pw) -> b (nh nw) (ph pw d)', ph=self.patch_size[0], pw=self.patch_size[1]))

        # Create positional embedding + modality embedding
        x_emb = repeat(self.pos_emb + self.mod_emb, '() n d -> b n d', b=B)

        d['x'] = x_patch
        d['emb'] = x_emb

        return d


class SequenceEmbEncoderEmbedding(nn.Module):
    """Adapter for sequence emb inputs, like T5-XXL, CLIP text embeddings.

    Args:
        max_length: Maximum number of tokens in the sequence
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 1D sin-cos positional embeddings
        padding_idx: Padding index for word embedding
        orig_emb_dim: Dimension of original embeddings
        bottleneck_dim: Dimension of bottleneck layer
        use_bottleneck: Set to True to use bottleneck layer
    """
    def __init__(self,
                 max_length: int,
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 max_sincos_pos_emb: int = 512,
                 padding_idx: int = 0,
                 orig_emb_dim: int = 4096,
                 bottleneck_dim: int = 64,
                 use_bottleneck: bool = False,
                 ):
        super().__init__()
        self.max_length = max_length
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.padding_idx = padding_idx
        self.max_sincos_pos_emb = max_sincos_pos_emb
        self.orig_emb_dim = orig_emb_dim
        self.use_bottleneck = use_bottleneck
        if self.use_bottleneck:
            self.bottleneck_dim = bottleneck_dim

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of embedding module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        if self.sincos_pos_emb:
            if self.max_length > self.max_sincos_pos_emb:
                raise ValueError(f"Max length ({self.max_length}) is greater than the number of posembs ({self.max_sincos_pos_emb}")
            pos_emb = build_1d_sincos_posemb(max_len=self.max_sincos_pos_emb, embed_dim=self.dim_tokens)[:self.max_length]
            self.register_buffer("pos_emb", pos_emb) # self.pos_emb is now a buffer for FSDP
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.max_length, self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding projection
        if self.use_bottleneck:
            self.emb_proj = nn.Sequential(
                nn.Linear(self.orig_emb_dim, self.bottleneck_dim),
                nn.Linear(self.bottleneck_dim, self.dim_tokens),
            )
        else:
            self.emb_proj = nn.Linear(self.orig_emb_dim, self.dim_tokens)


    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d):
        """
        Forward pass through embedding module, projecting original embeddings to the Transformer dimension.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict with at least the following keys:
                - 'tensor' (torch.Tensor): Input token sequence for each batch. Shape (B, L, E) where B is the batch size and L is the sequence length, and E is the dimension of the original embeddings.
                - 'input_mask' (torch.Tensor): Mask for valid tokens in the input sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, L, D) where D is the Transformer embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, L, D).
        """
        orig_emb = d['tensor']
        B = orig_emb.shape[0]
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'

        # Map to embedding
        x = self.emb_proj(orig_emb)

        expanded_pos_emb = repeat(self.pos_emb, "() n d -> b n d", b=B)
        # Input pos encoding
        input_mask = d['input_mask']
        input_pos_id = (~input_mask).int().cumsum(dim=1) - 1
        input_pos_id[input_mask] = 0
        input_pos_emb = torch.gather(expanded_pos_emb, dim=1, index=repeat(input_pos_id, "b n -> b n d", d=expanded_pos_emb.shape[2]))
        input_pos_emb[input_mask] = 0

        x_emb = input_pos_emb + self.mod_emb

        d['x'] = x
        d['emb'] = x_emb
        return d
    