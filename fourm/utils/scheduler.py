
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
# --------------------------------------------------------
# Based on DINO code base
# https://github.com/facebookresearch/dino
# --------------------------------------------------------
import numpy as np
import math

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0 or warmup_steps > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def constant_scheduler(base_value, epochs, niter_per_ep):
    return base_value * np.ones(epochs*niter_per_ep)


def inverse_sqrt_scheduler(base_value, final_value, epochs, niter_per_ep,  warmup_epochs=0, 
                           start_warmup_value=0, warmup_steps=-1, 
                           cooldown_epochs=0, cooldown_steps=-1, 
                           timescale=10_000):
    
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    
    cooldown_iters = cooldown_epochs * niter_per_ep
    if cooldown_steps > 0:
        cooldown_iters = cooldown_steps
    print("Set cooldown steps = %d" % cooldown_iters)
    
    # Warmup schedule
    if warmup_epochs > 0 or warmup_steps > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    else:
        warmup_schedule = np.array([])
    
    # Inverse square-root LR schedule
    iters = np.arange(epochs * niter_per_ep - warmup_iters - cooldown_iters)
    if base_value == final_value:
        schedule = base_value * np.ones(len(iters))
    else:
        schedule = base_value / np.sqrt((iters + timescale) / timescale)
    
    # Cooldown schedule
    if cooldown_epochs > 0 or cooldown_steps > 0:
        cooldown_schedule = np.linspace(schedule[-1], final_value, cooldown_iters)
    else:
        cooldown_schedule = np.array([])

    schedule = np.concatenate((warmup_schedule, schedule, cooldown_schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule