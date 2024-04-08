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
from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import ControlNetModel
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.controlnet import zero_module

from fourm.utils import to_2tuple
from .lm_models import create_model


class ControlNetAdapterEmbedding(nn.Module):

    def __init__(
        self,
        conditioning_embedding_channels,
        adapter,
        conditioning_channels=3,
    ):
        super().__init__()

        self.adapter_model = create_model(
            in_channels=conditioning_channels,
            output_type="stats",
        )
        self._load_adapter(adapter)

        self.conv_out = zero_module(
            nn.Conv2d(8, conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.adapter_model(quant=conditioning)

        embedding = self.conv_out(embedding)

        return embedding

    def _load_adapter(self, path):
        ckpt = torch.load(path)['model']
        for key in list(ckpt.keys()):
            if 'vq_model' in key or 'vae' in key:
                del ckpt[key]
        self.adapter_model.load_state_dict(ckpt)
        print("Loaded the adapter model")


class ControlNetConditioningEmbedding(nn.Module):

    def __init__(
        self,
        conditioning_embedding_channels,
        conditioning_channels = 3,
        block_out_channels = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding

    
class ControlnetCond(ModelMixin, ConfigMixin):
    def __init__(self, 
        in_channels, 
        cond_channels, 
        sd_pipeline, 
        image_size, 
        freeze_params=True,
        block_out_channels = (320, 640, 1280, 1280),
        conditioning_embedding_out_channels = (32, 32, 96, 256),
        pretrained_cn=False,
        enable_xformer=False,
        adapter=None,
        *args, 
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels

        self.sd_pipeline = sd_pipeline
        self.unet = sd_pipeline.unet
        self.text_encoder = sd_pipeline.text_encoder
        self.tokenizer = sd_pipeline.tokenizer

        if pretrained_cn:
            self.controlnet = ControlNetModel.from_unet(self.unet, conditioning_embedding_out_channels=conditioning_embedding_out_channels)
            self.controlnet.conditioning_channels = cond_channels
            self.controlnet.config.conditioning_channels = cond_channels
        else:
            self.controlnet = ControlNetModel(
                    in_channels=in_channels,
                    conditioning_channels=cond_channels,
                    block_out_channels=block_out_channels,
                    conditioning_embedding_out_channels=conditioning_embedding_out_channels,
                    *args,
                    **kwargs,
                )

        self.use_adapter = adapter is not None
        if adapter is not None:
            self.controlnet.controlnet_cond_embedding = ControlNetAdapterEmbedding(
                conditioning_embedding_channels=self.controlnet.config.block_out_channels[0],
                adapter=adapter,
                conditioning_channels=cond_channels,
            )
        else:
            self.controlnet.controlnet_cond_embedding = ControlNetConditioningEmbedding(
                conditioning_embedding_channels=self.controlnet.config.block_out_channels[0],
                block_out_channels=self.controlnet.config.conditioning_embedding_out_channels,
                conditioning_channels=cond_channels,
            )

        if enable_xformer:
            print('xFormer enabled')
            self.unet.enable_xformers_memory_efficient_attention()
            self.controlnet.enable_xformers_memory_efficient_attention()
        
        self.empty_str_encoding = nn.Parameter(self._encode_prompt(""), requires_grad=False)
        if freeze_params:
            self.freeze_params()

        self.sample_size = image_size // sd_pipeline.vae_scale_factor
        self.H, self.W = to_2tuple(self.sample_size)

    def forward(self,
                sample: torch.FloatTensor, # Shape (B, C, H, W),
                timestep: Union[torch.Tensor, float, int],
                encoder_hidden_states: torch.Tensor = None, # Shape (B, D_C, H_C, W_C)
                cond_mask: Optional[torch.BoolTensor] = None, # Boolen tensor of shape (B, H_C, W_C). True for masked out pixels,
                prompt = None,
                unconditional = False,
                cond_scale = 1.0,
                **kwargs):

        # Optionally mask out conditioning
        if cond_mask is not None:
            encoder_hidden_states = torch.where(cond_mask[:,None,:,:], 0.0, encoder_hidden_states)

        if not self.use_adapter:
            controlnet_cond = F.interpolate(encoder_hidden_states, (self.H, self.W), mode="nearest")
        else:
            controlnet_cond = F.interpolate(encoder_hidden_states, (self.H // 2, self.W // 2), mode="nearest")
        
        # encoder_hidden_states is the propmp embedding in the controlnet model, for now it's set to zeros.
        if prompt is None or unconditional:
            encoder_hidden_states = torch.cat([self.empty_str_encoding] * sample.shape[0])
        else:
            encoder_hidden_states = self._encode_prompt(prompt)

        down_block_res_samples, mid_block_res_sample = self.controlnet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
                conditioning_scale=cond_scale,
                return_dict=False,
            )

        # TODO not the most efficient way
        if unconditional:
            down_block_res_samples = [torch.zeros_like(s) for s in down_block_res_samples]
            controlnet_cond = torch.zeros_like(controlnet_cond)
        
        noise_pred = self.unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]

        return noise_pred
    
    def freeze_params(self):
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_params(self):
        for param in self.unet.parameters():
            param.requires_grad = True
        for param in self.text_encoder.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def _encode_prompt(self, prompt):

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(self.device)
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.device),
            attention_mask=attention_mask,
        )[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)

        return prompt_embeds


def controlnet(*args, **kwargs):
    return ControlnetCond(
            flip_sin_to_cos=True,
            freq_shift=0,
            down_block_types=
             ['CrossAttnDownBlock2D',
              'CrossAttnDownBlock2D',
              'CrossAttnDownBlock2D',
              'DownBlock2D'],
            only_cross_attention=False,
            block_out_channels=[320, 640, 1280, 1280],
            layers_per_block=2,
            downsample_padding=1,
            mid_block_scale_factor=1,
            act_fn='silu',
            norm_num_groups=32,
            norm_eps=1e-05,
            cross_attention_dim=768,
            attention_head_dim=8,
            num_attention_heads=None,
            use_linear_projection=False,
            class_embed_type=None,
            num_class_embeds=None,
            upcast_attention=False,
            resnet_time_scale_shift='default',
            projection_class_embeddings_input_dim=None,
            controlnet_conditioning_channel_order='rgb',
            conditioning_embedding_out_channels=[kwargs['cond_channels'], 32, 96, 256],
            global_pool_conditions=False,
            freeze_params=True,
            *args,
            **kwargs,
        )