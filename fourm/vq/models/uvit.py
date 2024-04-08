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
from typing import Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat

from diffusers.models.embeddings import (
    GaussianFourierProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.unet_2d_blocks import (
    DownBlock2D,
    UpBlock2D,
)
from diffusers.models.resnet import Downsample2D, Upsample2D
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin

# xFormers imports
try:
    from xformers.ops import memory_efficient_attention, unbind
    XFORMERS_AVAILABLE = True
except ImportError:
    print("xFormers not available")
    XFORMERS_AVAILABLE = False


def modulate(x, shift, scale):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.):
    """Sine-cosine positional embeddings as used in MoCo-v3

    Returns positional embedding of shape [B, H, W, D]
    """
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    pos_emb = rearrange(pos_emb, 'b (h w) d -> b d h w', h=h, w=w)
    return pos_emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, temb_dim=None, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, self.hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(self.hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        if temb_dim is not None:
            self.adaLN_modulation = nn.Linear(temb_dim, 2 * self.hidden_features)

    def forward(self, x, temb=None):
        x = self.fc1(x)
        x = self.act(x)
        
        # Shift and scale using time emb (see https://arxiv.org/abs/2301.11093)
        if hasattr(self, 'adaLN_modulation'):
            shift, scale = self.adaLN_modulation(F.silu(temb)).chunk(2, dim=-1)
            x = modulate(x, shift, scale)
        
        x = self.fc2(x)
        x = self.drop(x)
        return x

    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        if XFORMERS_AVAILABLE:
            q, k, v = unbind(qkv, 2)

            if mask is not None:
                # Wherever mask is True it becomes -infinity, otherwise 0
                mask = mask.to(q.dtype) * -torch.finfo(q.dtype).max

            x = memory_efficient_attention(q, k, v, attn_bias=mask)
            x = x.reshape([B, N, C])

        else:
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale

            if mask is not None:
                mask = mask.unsqueeze(1)
                attn = attn.masked_fill(mask, -torch.finfo(attn.dtype).max)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, dim_context=None, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        dim_context = dim_context or dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim_context, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, mask=None):
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads)

        if XFORMERS_AVAILABLE:
            k, v = unbind(kv, 2)

            if mask is not None:
                # Wherever mask is True it becomes -infinity, otherwise 0
                mask = mask.to(q.dtype) * -torch.finfo(q.dtype).max

            x = memory_efficient_attention(q, k, v, attn_bias=mask)
            x = x.reshape([B, N, C])

        else:
            q = q.permute(0, 2, 1, 3)
            kv = kv.permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                mask = rearrange(mask, "b n m -> b 1 n m")
                attn = attn.masked_fill(mask, -torch.finfo(attn.dtype).max)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
class Block(nn.Module):
    def __init__(self, dim, num_heads, temb_dim=None, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, temb_in_mlp=False, temb_after_norm=True, temb_gate=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, temb_dim=temb_dim if temb_in_mlp else None, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if temb_after_norm and temb_dim is not None:
            # adaLN modulation (see https://arxiv.org/abs/2212.09748)
            self.adaLN_modulation = nn.Linear(temb_dim, 4 * dim)
        if temb_gate and temb_dim is not None:
            # adaLN-Zero gate (see https://arxiv.org/abs/2212.09748)
            self.adaLN_gate = nn.Linear(temb_dim, 2 * dim)
            nn.init.zeros_(self.adaLN_gate.weight)
            nn.init.zeros_(self.adaLN_gate.bias)
        self.skip_linear = nn.Linear(2*dim, dim) if skip else None

    def forward(self, x, temb=None, mask=None, skip_connection=None):
        gate_msa, gate_mlp = self.adaLN_gate(F.silu(temb)).unsqueeze(1).chunk(2, dim=-1) if hasattr(self, 'adaLN_gate') else (1.0, 1.0)
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(F.silu(temb)).chunk(4, dim=-1) if hasattr(self, 'adaLN_modulation') else 4*[0.0]
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip_connection], dim=-1))
        x = x + gate_msa * self.drop_path(self.attn(modulate(self.norm1(x), shift_msa, scale_msa), mask))
        x = x + gate_mlp * self.drop_path(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp), temb))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, temb_dim=None, dim_context=None, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, temb_in_mlp=False, temb_after_norm=True, temb_gate=True):
        super().__init__()
        dim_context = dim_context or dim
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, dim_context=dim_context, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.query_norm = norm_layer(dim)
        self.context_norm = norm_layer(dim_context)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, temb_dim=temb_dim if temb_in_mlp else None, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if temb_after_norm and temb_dim is not None:
            # adaLN modulation (see https://arxiv.org/abs/2212.09748)
            self.adaLN_modulation = nn.Linear(temb_dim, 6 * dim)
        if temb_gate and temb_dim is not None:
            # adaLN-Zero gate (see https://arxiv.org/abs/2212.09748)
            self.adaLN_gate = nn.Linear(temb_dim, 3 * dim)
            nn.init.zeros_(self.adaLN_gate.weight)
            nn.init.zeros_(self.adaLN_gate.bias)
        self.skip_linear = nn.Linear(2*dim, dim) if skip else None

    def forward(self, x, context, temb=None, sa_mask=None, xa_mask=None, skip_connection=None):
        gate_msa, gate_mxa, gate_mlp = self.adaLN_gate(F.silu(temb)).unsqueeze(1).chunk(3, dim=-1) if hasattr(self, 'adaLN_gate') else (1.0, 1.0, 1.0)
        shift_msa, scale_msa, shift_mxa, scale_mxa, shift_mlp, scale_mlp = self.adaLN_modulation(F.silu(temb)).chunk(6, dim=-1) if hasattr(self, 'adaLN_modulation') else 6*[0.0]
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip_connection], dim=-1))
        x = x + gate_msa * self.drop_path(self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa), sa_mask))
        x = x + gate_mxa * self.drop_path(self.cross_attn(modulate(self.query_norm(x), shift_mxa, scale_mxa), self.context_norm(context), xa_mask))
        x = x + gate_mlp * self.drop_path(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp), temb))
        return x


class TransformerConcatCond(nn.Module):
    """UViT Transformer bottleneck that concatenates the condition to the input.
        
    Args:
        unet_dim: Number of channels in the last UNet down block.
        cond_dim: Number of channels in the condition.
        mid_layers: Number of Transformer layers.
        mid_num_heads: Number of attention heads.
        mid_dim: Transformer dimension.
        mid_mlp_ratio: Ratio of MLP hidden dim to Transformer dim.
        mid_qkv_bias: Whether to add bias to the query, key, and value projections.
        mid_drop_rate: Dropout rate.
        mid_attn_drop_rate: Attention dropout rate.
        mid_drop_path_rate: Stochastic depth rate.
        time_embed_dim: Dimension of the time embedding.
        hw_posemb: Size (side) of the 2D positional embedding.
        use_long_skip: Whether to use long skip connections.
          See https://arxiv.org/abs/2209.12152 for more details.
    """
    def __init__(
        self,
        unet_dim: int = 1024,
        cond_dim: int = 32,
        mid_layers: int = 12,
        mid_num_heads: int = 12,
        mid_dim: int = 768,
        mid_mlp_ratio: int = 4,
        mid_qkv_bias: bool = True,
        mid_drop_rate: float = 0.0,
        mid_attn_drop_rate: float = 0.0,
        mid_drop_path_rate: float = 0.0,
        time_embed_dim: int = 512,
        hw_posemb: int = 16,
        use_long_skip: bool = False,
    ):
        super().__init__()
        
        self.mid_pos_emb = build_2d_sincos_posemb(h=hw_posemb, w=hw_posemb, embed_dim=mid_dim)
        self.mid_pos_emb = nn.Parameter(self.mid_pos_emb, requires_grad=False)
        
        self.use_long_skip = use_long_skip
        if use_long_skip:
            assert mid_layers % 2 == 1, 'mid_layers must be odd when using long skip connection'

        dpr = [x.item() for x in torch.linspace(0, mid_drop_path_rate, mid_layers)] # stochastic depth decay rule
        self.mid_block = nn.ModuleList([
            Block(dim=mid_dim, temb_dim=time_embed_dim, num_heads=mid_num_heads, mlp_ratio=mid_mlp_ratio, qkv_bias=mid_qkv_bias,
                  drop=mid_drop_rate, attn_drop=mid_attn_drop_rate, drop_path=dpr[i], skip=i > mid_layers//2 and use_long_skip)
            for i in range(mid_layers)
        ])
        
        self.mid_cond_proj = nn.Linear(cond_dim, mid_dim)
        self.mid_proj_in = nn.Linear(unet_dim, mid_dim)
        self.mid_proj_out = nn.Linear(mid_dim, unet_dim)
                
        self.mask_token = nn.Parameter(torch.zeros(mid_dim), requires_grad=True)
        
        
    def forward(self, 
                x: torch.Tensor, 
                temb: torch.Tensor, 
                cond: torch.Tensor, 
                cond_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """TransformerConcatCond forward pass.
        
        Args:
            x: UNet features from the last down block of shape [B, C_mid, H_mid, W_mid].
            temb: Time embedding of shape [B, temb_dim].
            cond: Condition of shape [B, cond_dim, H_cond, W_cond]. If H_cond and W_cond are
              different from H_mid and W_mid, cond is interpolated to match the spatial size
              of x.
            cond_mask: Condition mask of shape [B, H_mid, W_mid]. If a mask is 
              defined, replaces masked-out tokens by a learnable mask-token. 
              Wherever cond_mask is True, the condition gets replaced by the mask token.

        Returns:
            Features of shape [B, C_mid, H_mid, W_mid] to pass to the UNet up blocks.
        """
        B, C_mid, H_mid, W_mid = x.shape

        # Rearrange and proj UNet features to sequence of tokens
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = self.mid_proj_in(x)

        # Rearrange and proj conditioning to sequence of tokens
        cond = F.interpolate(cond, (H_mid, W_mid)) # Interpolate if necessary
        cond = rearrange(cond, 'b d h w -> b (h w) d')
        cond = self.mid_cond_proj(cond)
        
        # If a mask is defined, replace masked-out tokens by a learnable mask-token
        # Wherever cond_mask is True, the condition gets replaced by the mask token 
        if cond_mask is not None:
            cond_mask = F.interpolate(cond_mask.unsqueeze(1).float(), (H_mid, W_mid), mode='nearest') > 0.5
            cond_mask = rearrange(cond_mask, 'b 1 h w -> b (h w)')
            cond[cond_mask] = self.mask_token.to(dtype=cond.dtype)
        
        x = x + cond

        # Interpolate and rearrange positional embedding to sequence of tokens
        mid_pos_emb = F.interpolate(self.mid_pos_emb, (H_mid, W_mid), mode='bicubic', align_corners=False)
        mid_pos_emb = rearrange(mid_pos_emb, 'b d h w -> b (h w) d')
        x = x + mid_pos_emb

        # Transformer forward pass with or without long skip connections
        if not self.use_long_skip:
            for blk in self.mid_block:
                x = blk(x, temb)
        else:
            skip_connections = []
            num_skips = len(self.mid_block) // 2
            for blk in self.mid_block[:num_skips]:
                x = blk(x, temb)
                skip_connections.append(x)
            x = self.mid_block[num_skips](x, temb)
            for blk in self.mid_block[num_skips+1:]:
                x = blk(x, temb, skip_connection=skip_connections.pop())

        x = self.mid_proj_out(x) # Project Transformer output back to UNet channels
        x = rearrange(x, 'b (h w) d -> b d h w', h=H_mid, w=W_mid) # Rearrange Transformer tokens to a spatial feature map for conv layers
        
        return x
    
class TransformerXattnCond(nn.Module):
    """UViT Transformer bottleneck that incroporates the condition via cross-attention.
        
    Args:
        unet_dim: Number of channels in the last UNet down block.
        cond_dim: Number of channels in the condition.
        mid_layers: Number of Transformer layers.
        mid_num_heads: Number of attention heads.
        mid_dim: Transformer dimension.
        mid_mlp_ratio: Ratio of MLP hidden dim to Transformer dim.
        mid_qkv_bias: Whether to add bias to the query, key, and value projections.
        mid_drop_rate: Dropout rate.
        mid_attn_drop_rate: Attention dropout rate.
        mid_drop_path_rate: Stochastic depth rate.
        time_embed_dim: Dimension of the time embedding.
        hw_posemb: Size (side) of the 2D positional embedding.
        use_long_skip: Whether to use long skip connections.
          See https://arxiv.org/abs/2209.12152 for more details.
    """
    def __init__(
        self,
        unet_dim: int = 1024,
        cond_dim: int = 32,
        mid_layers: int = 12,
        mid_num_heads: int = 12,
        mid_dim: int = 768,
        mid_mlp_ratio: int = 4,
        mid_qkv_bias: bool = True,
        mid_drop_rate: float = 0.0,
        mid_attn_drop_rate: float = 0.0,
        mid_drop_path_rate: float = 0.0,
        time_embed_dim: int = 512,
        hw_posemb: int = 16,
        use_long_skip: bool = False,
    ):
        super().__init__()

        self.mid_pos_emb = build_2d_sincos_posemb(h=hw_posemb, w=hw_posemb, embed_dim=mid_dim)
        self.mid_pos_emb = nn.Parameter(self.mid_pos_emb, requires_grad=False)
        
        self.use_long_skip = use_long_skip
        if use_long_skip:
            assert mid_layers % 2 == 1, 'mid_layers must be odd when using long skip connection'
        
        dpr = [x.item() for x in torch.linspace(0, mid_drop_path_rate, mid_layers)] # stochastic depth decay rule
        self.mid_block = nn.ModuleList([
            DecoderBlock(
                dim=mid_dim, temb_dim=time_embed_dim, num_heads=mid_num_heads, dim_context=cond_dim, 
                mlp_ratio=mid_mlp_ratio, qkv_bias=mid_qkv_bias, drop=mid_drop_rate, 
                attn_drop=mid_attn_drop_rate, drop_path=dpr[i], 
                skip=i > mid_layers//2 and use_long_skip
            )
            for i in range(mid_layers)
        ])
        
        self.mid_proj_in = nn.Linear(unet_dim, mid_dim)
        self.mid_proj_out = nn.Linear(mid_dim, unet_dim)        
        
    def forward(self, 
                x: torch.Tensor, 
                temb: torch.Tensor, 
                cond: torch.Tensor, 
                cond_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """TransformerXattnCond forward pass.
        
        Args:
            x: UNet features from the last down block of shape [B, C_mid, H_mid, W_mid].
            temb: Time embedding of shape [B, temb_dim].
            cond: Condition of shape [B, cond_dim, H_cond, W_cond].
            cond_mask: Condition cross-attention mask of shape [B, H_cond, W_cond]. 
              If a mask is defined, wherever cond_mask is True, the condition at that
              spatial location is not cross-attended to.

        Returns:
            Features of shape [B, C_mid, H_mid, W_mid] to pass to the UNet up blocks.
        """
        B, C_mid, H_mid, W_mid = x.shape

        # Rearrange and proj UNet features to sequence of tokens
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = self.mid_proj_in(x)

        # Rearrange conditioning to sequence of tokens
        cond = rearrange(cond, 'b d h w -> b (h w) d')

        # Interpolate and rearrange positional embedding to sequence of tokens
        mid_pos_emb = F.interpolate(self.mid_pos_emb, (H_mid, W_mid), mode='nearest')
        mid_pos_emb = rearrange(mid_pos_emb, 'b d h w -> b (h w) d')
        # Add UNet mid-block features and positional embedding
        x = x + mid_pos_emb

        # Prepare the conditioning cross-attention mask
        xa_mask = repeat(cond_mask, 'b h w -> b n (h w)', n=x.shape[1]) if cond_mask is not None else None

        # Transformer forward pass with or without long skip connections.
        # In each layer, cross-attend to the conditioning.
        if not self.use_long_skip:
            for blk in self.mid_block:
                x = blk(x, cond, temb, xa_mask=xa_mask)
        else:
            skip_connections = []
            num_skips = len(self.mid_block) // 2
            for blk in self.mid_block[:num_skips]:
                x = blk(x, cond, temb, xa_mask=xa_mask)
                skip_connections.append(x)
            x = self.mid_block[num_skips](x, cond, temb, xa_mask=xa_mask)
            for blk in self.mid_block[num_skips+1:]:
                x = blk(x, cond, temb, xa_mask=xa_mask, skip_connection=skip_connections.pop())

        x = self.mid_proj_out(x) # Project Transformer output back to UNet channels
        x = rearrange(x, 'b (h w) d -> b d h w', h=H_mid, w=W_mid) # Rearrange Transformer tokens to a spatial feature map for conv layers
        
        return x
    
                
class UViT(ModelMixin, ConfigMixin):
    """UViT model = Conditional UNet with Transformer bottleneck 
    blocks and optionalpatching.
    See https://arxiv.org/abs/2301.11093 for more details.
    
    Args:
        sample_size: Size of the input images.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        patch_size: Size of the input patching operation.
          See https://arxiv.org/abs/2207.04316 for more details.
        block_out_channels: Number of output channels of each UNet ResNet-block.
        layers_per_block: Number of ResNet blocks per UNet block.
        downsample_before_mid: Whether to downsample before the Transformer bottleneck.
        mid_layers: Number of Transformer blocks.
        mid_num_heads: Number of attention heads.
        mid_dim: Transformer dimension.
        mid_mlp_ratio: Transformer MLP ratio.
        mid_qkv_bias: Whether to use bias in the Transformer QKV projection.
        mid_drop_rate: Dropout rate of the Transformer MLP and attention output projection.
        mid_attn_drop_rate: Dropout rate of the Transformer attention.
        mid_drop_path_rate: Stochastic depth rate of the Transformer blocks.
        mid_hw_posemb: Size (side) of the Transformer positional embedding.
        mid_use_long_skip: Whether to use long skip connections in the Transformer blocks.
          See https://arxiv.org/abs/2209.12152 for more details.
        cond_dim: Dimension of the conditioning vector.
        cond_type: Type of conditioning. 
          'concat' for concatenation, 'xattn' for cross-attention.
        downsample_padding: Padding of the UNet downsampling convolutions.
        act_fn: Activation function.
        norm_num_groups: Number of groups in the UNet ResNet-block normalization.
        norm_eps: Epsilon of the UNet ResNet-block normalization.
        resnet_time_scale_shift: Time scale shift of the UNet ResNet-blocks.
        resnet_out_scale_factor: Output scale factor of the UNet ResNet-blocks.
        time_embedding_type: Type of the time embedding. 
          'positional' for positional, 'fourier' for Fourier.
        time_embedding_dim: Dimension of the time embedding.
        time_embedding_act_fn: Activation function of the time embedding.
        timestep_post_act: Activation function after the time embedding.
        time_cond_proj_dim: Dimension of the optional conditioning projection.
        flip_sin_to_cos: Whether to flip the sine to cosine in the time embedding.
        freq_shift: Frequency shift of the time embedding.
        res_embedding: Whether to perform original resolution conditioning.
          See SDXL https://arxiv.org/abs/2307.01952 for more details.
    """
    def __init__(self,
                 # UNet settings
                 sample_size: Optional[int] = None,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 patch_size: int = 4,
                 block_out_channels: Tuple[int] = (128, 256, 512),
                 layers_per_block: Union[int, Tuple[int]] = 2,
                 downsample_before_mid: bool = False,
                 
                 # Mid-block Transformer settings
                 mid_layers: int = 12,
                 mid_num_heads: int = 12,
                 mid_dim: int = 768,
                 mid_mlp_ratio: int = 4,
                 mid_qkv_bias: bool = True,
                 mid_drop_rate: float = 0.0,
                 mid_attn_drop_rate: float = 0.0,
                 mid_drop_path_rate: float = 0.0,
                 mid_hw_posemb: int = 32,
                 mid_use_long_skip: bool = False,
                 
                 # Conditioning settings
                 cond_dim: int = 32,
                 cond_type: str = 'concat',
                 
                 # ResNet blocks settings
                 downsample_padding: int = 1,
                 act_fn: str = "silu",
                 norm_num_groups: Optional[int] = 32,
                 norm_eps: float = 1e-5,
                 resnet_time_scale_shift: str = "default",
                 resnet_out_scale_factor: int = 1.0,
                      
                 # Time embedding settings
                 time_embedding_type: str = "positional",
                 time_embedding_dim: Optional[int] = None,
                 time_embedding_act_fn: Optional[str] = None,
                 timestep_post_act: Optional[str] = None,
                 time_cond_proj_dim: Optional[int] = None,
                 flip_sin_to_cos: bool = True,
                 freq_shift: int = 0,
                 
                 # Original resolution embedding settings
                 res_embedding: bool = False):

        super().__init__()

        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_dim = block_out_channels[-1]
        self.res_embedding = res_embedding

        # input patching
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=patch_size, padding=0, stride=patch_size
        )

        # time
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )

        if time_embedding_act_fn is None:
            self.time_embed_act = None
        elif time_embedding_act_fn == "swish":
            self.time_embed_act = lambda x: F.silu(x)
        elif time_embedding_act_fn == "mish":
            self.time_embed_act = nn.Mish()
        elif time_embedding_act_fn == "silu":
            self.time_embed_act = nn.SiLU()
        elif time_embedding_act_fn == "gelu":
            self.time_embed_act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {time_embedding_act_fn}")

        # original resolution embedding
        if res_embedding:
            if time_embedding_type == "fourier":
                self.h_proj = GaussianFourierProjection(
                    time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
                )
                self.w_proj = GaussianFourierProjection(
                    time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
                )
            elif time_embedding_type == "positional":
                self.height_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
                self.width_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)

            self.height_embedding = TimestepEmbedding(
                timestep_input_dim, time_embed_dim, act_fn=act_fn,
                post_act_fn=timestep_post_act, cond_proj_dim=time_cond_proj_dim,
            )
            self.width_embedding = TimestepEmbedding(
                timestep_input_dim, time_embed_dim, act_fn=act_fn,
                post_act_fn=timestep_post_act, cond_proj_dim=time_cond_proj_dim,
            )

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(block_out_channels)

        # down
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = DownBlock2D(
                num_layers=layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
                output_scale_factor=resnet_out_scale_factor,                
            )
            self.down_blocks.append(down_block)
        if downsample_before_mid:
            self.downsample_mid = Downsample2D(self.mid_dim, use_conv=True, out_channels=self.mid_dim)
            
        # mid
        if cond_type == 'concat':
            self.mid_block = TransformerConcatCond(
                unet_dim=self.mid_dim, cond_dim=cond_dim, mid_layers=mid_layers, mid_num_heads=mid_num_heads,
                mid_dim=mid_dim, mid_mlp_ratio=mid_mlp_ratio, mid_qkv_bias=mid_qkv_bias, 
                mid_drop_rate=mid_drop_rate, mid_attn_drop_rate=mid_attn_drop_rate, mid_drop_path_rate=mid_drop_path_rate,
                time_embed_dim=time_embed_dim, hw_posemb=mid_hw_posemb, use_long_skip=mid_use_long_skip, 
            )
        elif cond_type == 'xattn':
            self.mid_block = TransformerXattnCond(
                unet_dim=self.mid_dim, cond_dim=cond_dim, mid_layers=mid_layers, mid_num_heads=mid_num_heads,
                mid_dim=mid_dim, mid_mlp_ratio=mid_mlp_ratio, mid_qkv_bias=mid_qkv_bias, 
                mid_drop_rate=mid_drop_rate, mid_attn_drop_rate=mid_attn_drop_rate, mid_drop_path_rate=mid_drop_path_rate,
                time_embed_dim=time_embed_dim, hw_posemb=mid_hw_posemb, use_long_skip=mid_use_long_skip, 
            )
        else:
            raise ValueError(f"Unsupported cond_type: {cond_type}")
        
        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        if downsample_before_mid:
            self.upsample_mid = Upsample2D(self.mid_dim, use_conv=True, out_channels=self.mid_dim)
        
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_layers_per_block = list(reversed(layers_per_block))

        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = UpBlock2D(
                num_layers=reversed_layers_per_block[i] + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                output_scale_factor=resnet_out_scale_factor,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )

            if act_fn == "swish":
                self.conv_act = lambda x: F.silu(x)
            elif act_fn == "mish":
                self.conv_act = nn.Mish()
            elif act_fn == "silu":
                self.conv_act = nn.SiLU()
            elif act_fn == "gelu":
                self.conv_act = nn.GELU()
            else:
                raise ValueError(f"Unsupported activation function: {act_fn}")

        else:
            self.conv_norm_out = None
            self.conv_act = None

        self.conv_out = nn.ConvTranspose2d(
            block_out_channels[0], out_channels, kernel_size=patch_size, stride=patch_size
        )
        
        self.init_weights()

    def init_weights(self) -> None:
        """Weight initialization following MAE's initialization scheme"""

        for name, m in self.named_modules():
            # Handle already zero-init gates
            if "adaLN_gate" in name:
                continue
            # Handle ResNet gates that were not initialized by diffusers
            if "conv2" in name:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
                
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
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            # Embedding
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=self.init_std)
            # Conv2d
            elif isinstance(m, nn.Conv2d):
                if '.conv_in' in name or '.conv_out' in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        condition: torch.Tensor,
        cond_mask: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        orig_res: Optional[Union[torch.LongTensor, Tuple[int, int]]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """UViT forward pass.
        
        Args:
            sample: Noisy image of shape (B, C, H, W).
            timestep: Timestep(s) of the current batch.
            condition: Conditioning tensor of shape (B, C_cond, H_cond, W_cond). When concatenating 
              the condition, it is interpolated to the resolution of the transformer (H_mid, W_mid).
            cond_mask: Mask tensor of shape (B, H_mid, W_mid) when concatenating the condition
              to the transformer, and (B, H_cond, W_cond) when using cross-attention. True for 
              masked out / ignored regions.
            timestep_cond: Optional conditioning to add to the timestep embedding.
            orig_res: The original resolution of the image to condition the diffusion on. Ignored if None.
              See SDXL https://arxiv.org/abs/2307.01952 for more details.

        Returns:
            Diffusion objective target image of shape (B, C, H, W).
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True
            
        # 1. time
        timesteps = timestep
        is_mps = sample.device.type == "mps"
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        # 1.5 original resolution conditioning (see SDXL paper)
        if orig_res is not None and self.res_embedding:
            if not torch.is_tensor(orig_res):
                h_orig, w_orig = orig_res
                dtype = torch.int32 if is_mps else torch.int64
                h_orig = torch.tensor([h_orig], dtype=dtype, device=sample.device).expand(sample.shape[0])
                w_orig = torch.tensor([w_orig], dtype=dtype, device=sample.device).expand(sample.shape[0])
            else:
                h_orig, w_orig = orig_res[:,0], orig_res[:,1]

            h_emb = self.height_proj(h_orig).to(dtype=sample.dtype)
            w_emb = self.width_proj(w_orig).to(dtype=sample.dtype)

            emb = emb + self.height_embedding(h_emb)
            emb = emb + self.width_embedding(w_emb)

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples
        if hasattr(self, 'downsample_mid'):
            sample = self.downsample_mid(sample)
        
        # 4. mid
        sample = self.mid_block(sample, emb, condition, cond_mask)

        # 5. up
        if hasattr(self, 'upsample_mid'):
            sample = self.upsample_mid(sample)
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
            
            sample = upsample_block(
                hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
            )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


def uvit_b_p4_f16(**kwargs):
    return UViT(
        patch_size=4,
        block_out_channels=(128, 256),
        layers_per_block=2,
        downsample_before_mid=True,
        mid_layers=12,
        mid_num_heads=12,
        mid_dim=768,
        mid_mlp_ratio=4,
        mid_qkv_bias=True,
        **kwargs
    )

def uvit_l_p4_f16(**kwargs):
    return UViT(
        patch_size=4,
        block_out_channels=(128, 256),
        layers_per_block=2,
        downsample_before_mid=True,
        mid_layers=24,
        mid_num_heads=16,
        mid_dim=1024,
        mid_mlp_ratio=4,
        mid_qkv_bias=True,
        **kwargs
    )


def uvit_h_p4_f16(**kwargs):
    return UViT(
        patch_size=4,
        block_out_channels=(128, 256),
        layers_per_block=2,
        downsample_before_mid=True,
        mid_layers=32,
        mid_num_heads=16,
        mid_dim=1280,
        mid_mlp_ratio=4,
        mid_qkv_bias=True,
        **kwargs
    )

def uvit_b_p4_f16_longskip(**kwargs):
    return UViT(
        patch_size=4,
        block_out_channels=(128, 256),
        layers_per_block=2,
        downsample_before_mid=True,
        mid_layers=13,
        mid_num_heads=12,
        mid_dim=768,
        mid_mlp_ratio=4,
        mid_qkv_bias=True,
        mid_use_long_skip=True,
        **kwargs
    )

def uvit_l_p4_f16_longskip(**kwargs):
    return UViT(
        patch_size=4,
        block_out_channels=(128, 256),
        layers_per_block=2,
        downsample_before_mid=True,
        mid_layers=25,
        mid_num_heads=16,
        mid_dim=1024,
        mid_mlp_ratio=4,
        mid_qkv_bias=True,
        mid_use_long_skip=True,
        **kwargs
    )

def uvit_b_p4_f8(**kwargs):
    return UViT(
        patch_size=4,
        block_out_channels=(128, 256),
        layers_per_block=2,
        downsample_before_mid=False,
        mid_layers=12,
        mid_num_heads=12,
        mid_dim=768,
        mid_mlp_ratio=4,
        mid_qkv_bias=True,
        **kwargs
    )

def uvit_l_p4_f8(**kwargs):
    return UViT(
        patch_size=4,
        block_out_channels=(128, 256),
        layers_per_block=2,
        downsample_before_mid=False,
        mid_layers=24,
        mid_num_heads=16,
        mid_dim=1024,
        mid_mlp_ratio=4,
        mid_qkv_bias=True,
        **kwargs
    )

def uvit_b_p4_f16_extraconv(**kwargs):
    return UViT(
        patch_size=4,
        block_out_channels=(128, 256, 512),
        layers_per_block=2,
        downsample_before_mid=False,
        mid_layers=12,
        mid_num_heads=12,
        mid_dim=768,
        mid_mlp_ratio=4,
        mid_qkv_bias=True,
        **kwargs
    )

def uvit_l_p4_f16_extraconv(**kwargs):
    return UViT(
        patch_size=4,
        block_out_channels=(128, 256, 512),
        layers_per_block=2,
        downsample_before_mid=False,
        mid_layers=24,
        mid_num_heads=16,
        mid_dim=1024,
        mid_mlp_ratio=4,
        mid_qkv_bias=True,
        **kwargs
    )

