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
import numpy as np


def compute_codebook_usage(
        all_tokens: torch.LongTensor, 
        codebook_size: int = 16_384, 
        window_size: int = 65_536) -> float:
    """Computes the codebook usage for a given set of encoded tokens, by computing the 
    percentage of unique tokens in windows of a given size. The window size should be
    chosen as batch_size * sequence_length, where batch_size is recommended to be set
    to 256, and the sequence_length is the number of tokens per image. We follow
    ViT-VQGAN's approach of using batch_size 256. (https://arxiv.org/abs/2110.04627)

    Args:
        all_tokens: A tensor of shape (n_tokens, ) containing all the encoded tokens.
        codebook_size: The size of the codebook.
        window_size: The size of the window to compute the codebook usage in.

    Returns:
        The average codebook usage.
    """
    n_full_windows = all_tokens.shape[0] // window_size
    
    percentages = []
    for i, token_window in enumerate(torch.split(all_tokens, window_size)):
        if i < n_full_windows:
            usage_perc = len(np.unique(token_window)) / codebook_size
            percentages.append(usage_perc)
        else:
            break
            
    return np.mean(percentages)