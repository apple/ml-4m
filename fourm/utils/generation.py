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
import numpy as np
import math


def sample_to_batch(mod_dict, device, domains):
    mod_dict = {
        modality: {k: v.unsqueeze(0).to(device, non_blocking=True) for k, v in d.items()}
        for modality, d in mod_dict.items() if modality in domains
    }
    
    return mod_dict


def unbatch(tensor):
    return tensor.detach().squeeze(0).cpu()


def batch_to_sample(mod_dict, domains):
    mod_dict = {
        modality: {k: unbatch(v) for k, v in d.items()}
        for modality, d in mod_dict.items() if modality in domains
    }
    
    return mod_dict


def batch_to_device(mod_dict, device, domains):
    mod_dict = {
        modality: {k: v.to(device, non_blocking=True) for k, v in d.items()}
        for modality, d in mod_dict.items() if modality in domains
    }
    
    return mod_dict


def cosine_schedule(num_steps, total_tokens):
    iters = np.arange(num_steps)
    base_value = 1
    final_value = 0
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])
    schedule_tokens = [round(total_tokens * i) for i in (schedule[:-1] - schedule[1:])]
    schedule_tokens.append(total_tokens - sum(schedule_tokens))
    return np.array(schedule_tokens)


def linear_schedule(num_steps, total_tokens):
    schedule = np.linspace(0, total_tokens, num_steps + 1, dtype=int)
    schedule_tokens = np.diff(schedule)[::-1]
    schedule_tokens.sort()  # Sorts the array in ascending order.
    schedule_tokens = schedule_tokens[::-1]  # Reverses the array to descending order.
    return np.trim_zeros(schedule_tokens, 'b')  # Trims trailing zeros.


def continue_schedule(schedule, num_current_tokens):
    schedule_cumsum = np.cumsum(schedule)
    keep_mask = schedule_cumsum > num_current_tokens
    diff = schedule_cumsum[keep_mask][0] - num_current_tokens
    new_schedule = schedule[keep_mask]
    new_schedule[0] = diff
    return new_schedule


def decreasing_temp_schedule(max, min, token_schedule):
    schedule_cumsum = np.cumsum(token_schedule) / np.sum(token_schedule)
    temp_schedule = np.array([min + (max - min) * (1 - s) for s in schedule_cumsum])
    return temp_schedule


def onex_temp_schedule(max_t, min_t, token_schedule, power=0.5, min_linspace=1, max_linspace=100):
    """Abitrary temperature schedule for one over x"""
    x = np.linspace(min_linspace, max_linspace, num=sum(token_schedule))
    y = 1/(x**power)
    y = y - min(y)
    y = y / max(y)
    unscaled_schedule = y
    schedule_cumsum = np.cumsum(token_schedule) / np.sum(token_schedule)
    unscaled_schedule = [(1 - cs) * us for us, cs in zip(unscaled_schedule, schedule_cumsum)]

    temp_schedule = np.array([min_t + (max_t - min_t) * s for s in unscaled_schedule]).clip(min=1e-9)
    return temp_schedule


def linear_temp_schedule(temp, token_schedule):
    """ Temperature that decays the temperature inversely proportional to the token schedule. """
    return np.concatenate([np.array([temp * 1.0]), (temp * (token_schedule.sum() - token_schedule.cumsum()) / token_schedule.sum())[:-1]]).clip(min=1e-9)
