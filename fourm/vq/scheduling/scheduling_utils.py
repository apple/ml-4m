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
import numpy as np
import torch


def enforce_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    """Scales the noise schedule betas so that last time step has zero SNR.
    See https://arxiv.org/abs/2305.08891

    Args:
        betas: the initial diffusion noise schedule betas

    Returns:
        The diffusion noise schedule betas with the last time step having zero SNR
    """

    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    # Shift so last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    # Scale so first timestep is back to old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
        alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
    """Create a beta schedule that discretizes the given alpha_t_bar function, 
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to 
    the cumulative product of (1-beta) up to that part of the diffusion process.

    Args:
        num_diffusion_timesteps: the number of betas to produce.
        max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        The betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def scaled_cosine_alphas(num_diffusion_timesteps: int, noise_shift: float = 1.0) -> torch.Tensor:
    """Shifts a cosine noise schedule by a specified amount in log-SNR space.

    noise_shift = 1.0 corresponds to the standard cosine noise schedule.
    0 < noise_shift < 1.0 corresponds to a less noisy schedule (better 
    suited if the conditioning is highly informative, e.g. low-res images).
    noise_shift > 1.0 corresponds to a more noisy schedule (better suited 
    if the conditioning is not as informative, e.g. captions).

    See https://arxiv.org/abs/2305.18231

    Args:
        num_diffusion_timesteps: the number of diffusion timesteps.
        noise_shift: the amount to shift the noise schedule by in log-SNR space.

    Returns:
        The alphas_cumprod used by the diffusion noise scheduler
    """
    t = torch.linspace(0, 1, num_diffusion_timesteps).to(torch.float64)
    log_snr = -2 * (torch.tan(torch.pi * t / 2).log() + np.log(noise_shift))
    log_snr = log_snr.clamp(-15,15).float()
    alphas_cumprod = log_snr.sigmoid()
    alphas_cumprod[-1] = 0.0
    return alphas_cumprod