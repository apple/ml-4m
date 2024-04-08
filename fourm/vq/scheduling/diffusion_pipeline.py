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
import torch
from diffusers import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from tqdm import tqdm

from fourm.utils import to_2tuple


def rescale_noise_cfg(noise_cfg, noise_pred_conditional, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_conditional.std(dim=list(range(1, noise_pred_conditional.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class PipelineCond(DiffusionPipeline):
    """Pipeline for conditional image generation.

    This model inherits from `DiffusionPipeline`. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        model: The conditional diffusion model.
        scheduler: A diffusion scheduler, e.g. see scheduling_ddpm.py
    """
    def __init__(self, model: torch.nn.Module, scheduler: SchedulerMixin):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, 
                 cond: torch.Tensor, 
                 generator: Optional[torch.Generator] = None, 
                 timesteps: Optional[int] = None,
                 guidance_scale: float = 0.0, 
                 guidance_rescale: float = 0.0,
                 image_size: Optional[Union[Tuple[int, int], int]] = None, 
                 verbose: bool = True,
                 scheduler_timesteps_mode: str = 'trailing',
                 orig_res: Optional[Union[torch.LongTensor, Tuple[int, int]]] = None,
                 **kwargs) -> torch.Tensor:
        """The call function to the pipeline for conditional image generation.

        Args:
            cond: The conditional input to the model.
            generator: A torch.Generator to make generation deterministic.
            timesteps: The number of denoising steps. More denoising steps usually lead to a higher 
              quality image at the expense of slower inference. Defaults to the number of training 
              timesteps if not given.
            guidance_scale: The scale of the classifier-free guidance. If set to 0.0, no guidance is used.
            guidance_rescale: Rescaling factor to fix the variance when using guidance scaling.
            image_size: The size of the image to generate. If not given, the default training size 
              of the model is used.
            verbose: Whether to show a progress bar.
            scheduler_timesteps_mode: The mode to use for DDIMScheduler. One of `trailing`, `linspace`, 
              `leading`. See https://arxiv.org/abs/2305.08891 for more details.
            orig_res: The original resolution of the image to condition the diffusion on. Ignored if None.
              See SDXL https://arxiv.org/abs/2307.01952 for more details.

        Returns:
            The generated image.
        """

        timesteps = self.scheduler.config.num_train_timesteps if timesteps is None else timesteps
        batch_size, _, _, _ = cond.shape
        
        # Sample gaussian noise to begin loop
        image_size = self.model.sample_size if image_size is None else image_size
        image_size = to_2tuple(image_size)
        image = torch.randn(
            (batch_size, self.model.in_channels, image_size[0], image_size[1]),
            generator=generator,
        )
        image = image.to(self.model.device)

        do_cfg = callable(guidance_scale) or  guidance_scale > 1.0

        # Set step values
        self.scheduler.set_timesteps(timesteps, mode=scheduler_timesteps_mode)

        if verbose:
            pbar = tqdm(total=len(self.scheduler.timesteps))

        for t in self.scheduler.timesteps:
            # 1. Predict noise model_output
            model_output = self.model(image, t, cond, orig_res=orig_res, **kwargs)
            
            if do_cfg:
                model_output_uncond = self.model(image, t, cond, unconditional=True, **kwargs) # TODO: is there a better way to get unconditional output?

                if callable(guidance_scale):
                    guidance_scale_value = guidance_scale(t/self.scheduler.config.num_train_timesteps)
                else:
                    guidance_scale_value = guidance_scale
                model_output_cfg = model_output_uncond + guidance_scale_value * (model_output - model_output_uncond)

                if guidance_rescale > 0.0:
                    model_output = rescale_noise_cfg(model_output_cfg, model_output, guidance_rescale=guidance_rescale)
                else:
                    model_output = model_output_cfg

            # 2. Compute previous image: x_t -> t_t-1
            with torch.cuda.amp.autocast(enabled=False):
                image = self.scheduler.step(model_output.float(), t, image, generator=generator).prev_sample

            if verbose:
                pbar.update()
        if verbose:
            pbar.close()
            
        return image
    
