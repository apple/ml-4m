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
from typing import Optional
from torch import nn
from einops import rearrange


class BottleneckBlock(nn.Module):
    def __init__(self, thin, wide):
        super(BottleneckBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(thin, wide), 
            nn.GELU(), 
            nn.Linear(wide, thin)
        )

    def forward(self, x):
        out = self.block(x)

        return out


class StandardMLP(nn.Module):
    def __init__(self, dim_in, dim_out, widths):
        super(StandardMLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.widths = widths
        self.linear_in = nn.Linear(self.dim_in, self.widths[0])
        self.linear_out = nn.Linear(self.widths[-1], self.dim_out)
        self.layers = []
        self.layer_norms = []
        for i in range(len(self.widths) - 1):
            self.layers.append(nn.Linear(self.widths[i], self.widths[i + 1]))
            self.layer_norms.append(nn.LayerNorm(widths[i + 1]))

        self.layers = nn.ModuleList(self.layers)
        self.layernorms = nn.ModuleList(self.layer_norms)

    def forward(self, x):
        # If x is an image, apply MLP point-wise to each token/pixel
        if x.ndim == 4:
            _, _, H, W = x.shape
            x = rearrange(x, 'b d h w -> b (h w) d')
            x_is_image = True
        else:
            x_is_image = False

        z = self.linear_in(x)
        for layer, norm in zip(self.layers, self.layer_norms):
            z = norm(z)
            z = layer(z)

        out = self.linear_out(z)

        # If x was an image, rearrange back to image
        if x_is_image:
            out = rearrange(out, 'b (h w) d -> b d h w', h=H, w=W)

        return out


class BottleneckMLP(nn.Module):
    def __init__(self, dim_in, dim_out, block_dims):
        super(BottleneckMLP, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.block_dims = block_dims

        self.linear_in = nn.Linear(self.dim_in, self.block_dims[0][1])
        self.linear_out = nn.Linear(self.block_dims[-1][1], self.dim_out)
        blocks = []
        layernorms = []

        for block_dim in self.block_dims:
            wide, thin = block_dim
            blocks.append(BottleneckBlock(thin=thin, wide=wide))
            layernorms.append(nn.LayerNorm(thin))

        self.blocks = nn.ModuleList(blocks)
        self.layernorms = nn.ModuleList(layernorms)

    def forward(self, x):
        # If x is an image, apply MLP point-wise to each token/pixel
        if x.ndim == 4:
            _, _, H, W = x.shape
            x = rearrange(x, 'b d h w -> b (h w) d')
            x_is_image = True
        else:
            x_is_image = False

        x = self.linear_in(x)

        for block, norm in zip(self.blocks, self.layernorms):
            x = x + block(norm(x))

        out = self.linear_out(x)

        # If x was an image, rearrange back to image
        if x_is_image:
            out = rearrange(out, 'b (h w) d -> b d h w', h=H, w=W)

        return out


def build_mlp(model_id: str = "BottleneckMLP/B_6-Wi_1024", 
              dim_in: Optional[int] = None, 
              dim_out: Optional[int] = None, 
              **kwargs) -> nn.Module:
    """Constructs an MLP model from a model ID string, see
    "Scaling MLPs: A Tale of Inductive Bias" (https://arxiv.org/abs/2306.13575).
    
    Args:
        model_id: Model ID string. E.g. "BottleneckMLP/B_6-Wi_1024".
          See https://arxiv.org/abs/2306.13575 for options and details.
        dim_in: Input dimensionality. If None, defaults to MLP dimension.
        dim_out: Output dimensionality. If None, defaults to MLP dimension.

    Returns:
        MLP model.
    """
    model, architecture = model_id.split("/")
    assert model in ["BottleneckMLP", "MLP"], f"Model {model} not supported."

    sep = architecture.split("-")
    num_blocks = int(sep[0].split("_")[1])
    thin = int(sep[1].split("_")[1])

    # If dim_in and dim_out are not specified, use MLP dim
    dim_in = dim_in or thin
    dim_out = dim_out or thin

    if len(sep) == 3:
        expansion_factor = int(sep[2].split("_")[1])
    else:
        expansion_factor = 4

    if model == "BottleneckMLP":
        blocks = [[expansion_factor * thin, thin] for _ in range(num_blocks)]

        return BottleneckMLP(
            dim_in=dim_in,
            dim_out=dim_out,
            block_dims=blocks,
        )
    elif model == "MLP":
        blocks = [thin for _ in range(num_blocks)]

        return StandardMLP(
            dim_in=dim_in,
            dim_out=dim_out,
            widths=blocks,
        )