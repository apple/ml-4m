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
"""
    lm: latent mapping
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.unet_2d_blocks import UNetMidBlock2D, get_up_block


FREEZE_MODULES = ['encoder', 'quant_proj', 'quantize', 'cls_emb']

class Token2VAE(nn.Module):
    def __init__(
        self,
        in_channels=32,
        output_type="stats",
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D",),
        block_out_channels=(256, 512),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        vq_model=None,
        vae=None,
    ):
        super().__init__()

        assert output_type in ["stats", "sample"], "`output_type` can be either of 'stats' or 'sample'"
        self.output_type = output_type
        out_channels = 4 if output_type == "sample" else 8

        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
                resnet_time_scale_shift="group",
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.vq_model = vq_model        
        self.vae = vae

    @torch.no_grad()
    def vae_encode(self, x):
        assert self.vae is not None, "VAE is not initialized"
        z = self.vae.encode(x).latent_dist
        if self.output_type == "sample":
            z = z.sample()
        else:
            z = torch.cat((z.mean, z.std), dim=1)
        z = z * self.vae.config.scaling_factor
        return z

    @torch.no_grad()
    def vae_decode(self, x, clip=True):
        assert self.vae is not None, "VAE is not initialized"
        x = self.sample(x)
        x = self.vae.decode(x / self.vae.config.scaling_factor).sample
        if clip:
            x = torch.clip(x, min=-1, max=1)
        return x

    def sample(self, x):
        if x.shape[1] == 4:
            return x
        mean, std = x.chunk(2, dim=1)
        x = mean + std * torch.randn_like(std)
        return x

    def forward(self, quant=None, image=None):
        
        if quant is None: 
            assert image is not None, "Neither of `quant` or `image` are provided"
            assert self.vq_model is not None, "VQ encoder is not initialized"
            with torch.no_grad():
                quant, _, _ = self.vq_model.encode(image)

        x = self.conv_in(quant)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        # middle
        x = self.mid_block(x)
        x = x.to(upscale_dtype)

        # up
        for up_block in self.up_blocks:
            x = up_block(x)

        # post-process
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x

def create_model(
    in_channels=32,
    output_type="stats",
    vq_model=None,
    vae=None,
):
    return Token2VAE(
        in_channels=in_channels,
        output_type=output_type,
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D",),
        block_out_channels=(256, 512),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        vq_model=vq_model,
        vae=vae,
    )