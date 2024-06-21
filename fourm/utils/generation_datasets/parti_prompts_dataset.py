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
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class PartiPromptsDataset(Dataset):
    """
    Parti Prompts caption dataset. 
    
    Args:
        text_tokenizer (tokenizers.Tokenizer): The tokenizer to use for encoding the captions.
        max_length (int): The maximum sequence length of the captions.
        parti_prompts_csv (str): The path to the Parti Prompts dataset.
    """
    def __init__(self, text_tokenizer, max_length=128, parti_prompts_csv='fourm/utils/generation_datasets/PartiPrompts.tsv', parti_prompts_t5_embs=None, llm_embedder=None):
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length
        self.parti_prompts = pd.read_csv(parti_prompts_csv, sep='\t')

        self.pad_id = text_tokenizer.token_to_id("[PAD]")
        self.eos_id = text_tokenizer.token_to_id("[EOS]")
        if parti_prompts_t5_embs is not None:
            # T5 Embeddings are saved as a numpy array, so we need to load it
            self.t5_embs = np.load(parti_prompts_t5_embs)['emb']
            self.t5_masks = np.load(parti_prompts_t5_embs)['mask_valid']
            self.llm_embedder = None
        elif llm_embedder is not None:
            self.t5_embs = None
            self.llm_embedder = llm_embedder
        else:
            self.t5_embs = None
            self.llm_embedder = None

    def __getitem__(self, index):
        text = self.parti_prompts.Prompt[index]
        seq_ids = self.text_tokenizer.encode(text).ids + [self.eos_id]

        tensor = torch.ones(self.max_length, dtype=torch.int) * self.pad_id
        tensor[:len(seq_ids)] = torch.tensor(seq_ids, dtype=torch.int)

        out = {}
        out['caption'] = {'tensor': tensor}

        if self.t5_embs is not None:

            t5_emb = torch.tensor(self.t5_embs[index], dtype=torch.float32)
            t5_emb = pad_or_truncate(t5_emb, self.max_length)

            t5_mask = torch.tensor(self.t5_masks[index], dtype=torch.bool)
            t5_mask = pad_or_truncate(t5_mask, self.max_length)

            ascii_tensor = text_to_tensor(text, max_length=self.max_length * 10) # Save ASCII as tensor

            out['t5_caption'] = {
                'tensor': t5_emb,
                'mask_valid': t5_mask,
                'ascii_tensor': ascii_tensor,
            }
        elif self.llm_embedder is not None:
            t5_emb, _, t5_mask = self.llm_embedder.get_text_embeddings([text])
            t5_emb = pad_or_truncate(t5_emb.squeeze(0), self.max_length)
            t5_mask = pad_or_truncate(t5_mask.bool().squeeze(0), self.max_length)
            ascii_tensor = text_to_tensor(text, max_length=self.max_length * 10) # Save ASCII as tensor

            out['t5_caption'] = {
                'tensor': t5_emb,
                'mask_valid': t5_mask,
                'ascii_tensor': ascii_tensor,
            }

        return out
    
    def __len__(self):
        return len(self.parti_prompts)
    

def pad_or_truncate(tensor, fixed_length, padding_value=0):
    current_length = tensor.shape[0]
    
    if current_length < fixed_length:
        # Calculate padding sizes for all dimensions, but only pad along dim=0
        padding_sizes = [0] * 2 * len(tensor.shape)
        padding_sizes[1] = fixed_length - current_length
        return torch.nn.functional.pad(tensor, padding_sizes, 'constant', padding_value)
    else:
        return tensor[:fixed_length]
    
def text_to_tensor(text, max_length=None):
    """Converts plaintext to a tensor with optional padding."""
    ascii_values = [ord(c) for c in text]
    if max_length:
        while len(ascii_values) < max_length:
            ascii_values.append(0)  # Using 0 as the padding value
    return torch.tensor(ascii_values, dtype=torch.int)


def tensor_to_text(tensor):
    """Converts tensor back to plaintext. Assumes padding with zeros."""
    ascii_values = tensor.tolist()
    return ''.join(chr(val) for val in ascii_values if val != 0)
