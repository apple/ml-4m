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
import torch

def convert_samples_to_mod_dict(samples, input_mod, target_mod, num_input_tokens, num_target_tokens):
    """Converts a sample (e.g. a batch of RGB images) to a mod dict that can be passed directly to FourM.
    Assumes both the input modality and target modality are dense tasks.
    """

    B = samples.shape[0]
    device = samples.device

    if input_mod == target_mod:
        assert(num_input_tokens == num_target_tokens)
        mod_dict = {
            input_mod: {
                'tensor': samples,
                'input_mask': torch.zeros((B, num_input_tokens), dtype=torch.bool, device=device),
                'target_mask': torch.zeros((B, num_target_tokens), dtype=torch.bool, device=device),
                'decoder_attention_mask': torch.zeros((B, num_target_tokens), dtype=torch.int, device=device),
            },
        }
        mod_dict[input_mod]['decoder_attention_mask'][:, 0] = num_target_tokens

    else:
        mod_dict = {
            input_mod: {
                'tensor': samples,
                'input_mask': torch.zeros((B, num_input_tokens), dtype=torch.bool, device=samples.device),
                'target_mask': torch.ones((B, num_input_tokens), dtype=torch.bool, device=samples.device),
                'decoder_attention_mask': torch.zeros((B, num_input_tokens), dtype=torch.int, device=samples.device),
            },
            target_mod: {
                'tensor': torch.zeros((B, num_target_tokens), dtype=torch.long, device=samples.device),
                'input_mask': torch.ones((B, num_target_tokens), dtype=torch.bool, device=samples.device),
                'target_mask': torch.zeros((B, num_target_tokens), dtype=torch.bool, device=samples.device),
                'decoder_attention_mask': torch.ones((B, num_target_tokens), dtype=torch.int, device=samples.device),
            },
        }
        mod_dict[target_mod]['decoder_attention_mask'][:, 0] = num_target_tokens

    return mod_dict
