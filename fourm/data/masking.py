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
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tokenizers import Tokenizer
from torch.distributions import Dirichlet

from fourm.data.modality_transforms import get_transform_key
from fourm.utils import to_2tuple
from fourm.utils.tokenizer import get_sentinel_to_id_mapping


def sample_cosine(min_val: float = 0, max_val: float =1) -> float:
    """Sample a value from a cosine distribution between min_val and max_val

    Args:
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Sampled value
    """

    return min_val + 0.5 * (max_val - min_val) * (1 + math.cos(math.pi * random.uniform(0, 1)))


def sample_uniform(min_val: float = 0, max_val: float =1) -> float:
    """Sample a value from a uniform distribution between min_val and max_val

    Args:
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Sampled value
    """

    return random.uniform(min_val, max_val)


def simple_span_masking(sequence: List[int], sentinel_to_id: Dict[int, int], keep_prob: float) -> Tuple[List[int], List[int]]:
    """Span masking for a sequence

    Args:
        sequence: Sequence to mask
        sentinel_to_id: Mapping from sentinel to id
        keep_prob: Probability of keeping a token

    Returns:
        Masked input sequence and masked target sequence
    """
    sequence_length = len(sequence)
    # 0 for keep, 1 for mask
    masks = torch.where(torch.rand(sequence_length) <= keep_prob, 0, 1).bool().tolist()

    input_sequence = []
    target_sequence = []

    prev_mask = False
    sentinel_count = 0
    for token, mask in zip(sequence, masks):
        if mask:
            if not prev_mask:
                sentinel_count += 1
                input_sequence.append(sentinel_to_id[sentinel_count])
                target_sequence.append(sentinel_to_id[sentinel_count])
            prev_mask = True
            target_sequence.append(token)
        else:
            prev_mask = False
            input_sequence.append(token)

    target_sequence.append(sentinel_to_id[sentinel_count + 1])
    return input_sequence, target_sequence


def chunk_span_masking(sequence_chunks: List[List[int]], sentinel_to_id: Dict[int, int], keep_prob: float) -> Tuple[List[int], List[int]]:
    """Span masking where masking is performed at the chunk level.

    Args:
        sequence_chunks: Sequence chunks to mask
        sentinel_to_id: Mapping from sentinel to id
        keep_prob: Probability of keeping a token

    Returns:
        Masked input sequence and masked target sequence
    """
    chunk_length = len(sequence_chunks)
    # 0 for keep, 1 for mask
    masks = torch.where(torch.rand(chunk_length) <= keep_prob, 0, 1).bool().tolist()

    input_sequence = []
    target_sequence = []

    prev_mask = False
    sentinel_count = 0
    for chunk, mask in zip(sequence_chunks, masks):
        if mask:
            if not prev_mask:
                sentinel_count += 1
                input_sequence.append(sentinel_to_id[sentinel_count])
                target_sequence.append(sentinel_to_id[sentinel_count])
            prev_mask = True
            target_sequence.extend(chunk)
        else:
            prev_mask = False
            input_sequence.extend(chunk)

    target_sequence.append(sentinel_to_id[sentinel_count + 1])
    return input_sequence, target_sequence



class UnifiedMasking(object):
    def __init__(self,
                 modality_info: Dict,
                 text_tokenizer: Optional[Tokenizer],
                 input_tokens_range: Union[int, Tuple[int, int]],
                 target_tokens_range: Optional[Union[int, Tuple[int, int]]],
                 max_tries: int = 100,
                 sampling_weights: Optional[List[float]] = None,):
        """Performs masking on a dict of modalities (both image based and sequence based modalities)

        Args:
            modality_info: Dict with the modalities and their corresponding information
            text_tokenizer: Tokenizer to use for text modalities
            input_tokens_range: Range of number of tokens to mask in the input
            target_tokens_range: Range of number of tokens to mask in the target
            max_tries: Maximum number of tries to find a valid token budgets
            sampling_weights: Sampling weights for the mixture of Dirichlet distributions
        """
        self.input_tokens_range = to_2tuple(input_tokens_range)
        self.target_tokens_range = to_2tuple(target_tokens_range) if target_tokens_range is not None else None
        self.modality_info = modality_info
        self.num_modalities = len(modality_info)
        self.max_tries = max_tries
        self.min_tokens = torch.tensor([mod['min_tokens'] for mod in modality_info.values()])
        self.max_tokens = torch.tensor([mod['max_tokens'] for mod in modality_info.values()])
        self.mod_is_img = torch.tensor([mod['type'] == 'img' for mod in modality_info.values()])

        # Dirichlet sampling (supports a mixture of multiple Dirichlet distributions)
        eps = 1e-9
        input_alphas = torch.tensor([mod["input_alphas"] for mod in modality_info.values()])
        input_alphas = rearrange(input_alphas, "nmod nmix -> nmix nmod")
        self.input_dirichlets = [Dirichlet(torch.clamp(input_alpha, min=eps)) for input_alpha in input_alphas]
        target_alphas = torch.tensor([mod["target_alphas"] for mod in modality_info.values()])
        target_alphas = rearrange(target_alphas, "nmod nmix -> nmix nmod")
        self.target_dirichlets = [Dirichlet(torch.clamp(target_alpha, min=eps)) for target_alpha in target_alphas]
        assert(len(self.input_dirichlets) == len(self.target_dirichlets))
        self.num_dirichlets = len(self.input_dirichlets)
        if sampling_weights is not None:
            assert len(sampling_weights) == self.num_dirichlets
            self.sampling_weights = torch.tensor(sampling_weights)
        else:
            self.sampling_weights = None

        self.text_tokenizer = text_tokenizer
        self.keep_prob_decay_factor = 0.9
        self.sentinel_to_id = get_sentinel_to_id_mapping(text_tokenizer)
        self.sentinel_ids = set(self.sentinel_to_id.values())
        self.pad_id = text_tokenizer.token_to_id("[PAD]")
        self.eos_id = text_tokenizer.token_to_id("[EOS]")

    def input_token_budget(self, num_input_tokens, dir_idx=0):
        """Sample a token budget for the input

        Args:
            num_input_tokens: Number of tokens in the input

        Returns:
            Token budget for the input
        """
        # Get the number of tokens for each modality
        for i in range(self.max_tries):
            input_token_budget = (self.input_dirichlets[dir_idx].sample() * num_input_tokens).floor().int()
            diff = num_input_tokens - input_token_budget.sum()
            # Adds the remaining tokens by sampling from the Dirichlet and taking the argmax
            # This avoids adding tokens to modalities that shouldn't be sampled (i.e. with alphas ~=0)
            input_token_budget += torch.bincount(self.input_dirichlets[dir_idx].sample_n(diff).argmax(dim=-1), minlength=len(input_token_budget))

            # If token budget is over max tokens for a given modality, set it to max
            input_token_budget = torch.clamp(input_token_budget, max=self.max_tokens)

            if (input_token_budget >= self.min_tokens).all():
                return input_token_budget.tolist()

        print(f"More than max tries for input!")
        return input_token_budget.tolist()

    def target_token_budget(self, input_token_budget, num_target_tokens, dir_idx=0):
        """Sample a token budget for the target

        Args:
            input_token_budget: Token budget for the input
            num_target_tokens: Number of tokens in the target

        Returns:
            Token budget for the target
        """
        # We don't reduce the number of tokens for sequence based tasks
        max_tokens_remaining = torch.where(self.mod_is_img, self.max_tokens - torch.tensor(input_token_budget), self.max_tokens)
        max_tokens_remaining = torch.max(self.min_tokens, max_tokens_remaining)
        for i in range(self.max_tries):
            target_token_budget = (self.target_dirichlets[dir_idx].sample() * num_target_tokens).floor().int()
            diff = num_target_tokens - target_token_budget.sum()
            # Adds the remaining tokens by sampling from the Dirichlet and taking the argmax
            # This avoids adding tokens to modalities that shouldn't be sampled (i.e. with alphas ~=0)
            target_token_budget += torch.bincount(self.target_dirichlets[dir_idx].sample_n(diff).argmax(dim=-1), minlength=len(target_token_budget))

            # If token budget is over max tokens for a given modality, set it to max
            target_token_budget = torch.clamp(target_token_budget, max=max_tokens_remaining)

            if (target_token_budget >= self.min_tokens).all():
                return target_token_budget.tolist()

        print(f"More than max tries for target!")
        return target_token_budget.tolist()

    def image_mask(self, tensor: torch.Tensor, num_tokens: int, input_budget: int, target_budget: int):
        """Applies input and target masking to an image tensor

        Args:
            tensor: Image tensor
            num_tokens: Number of tokens in the tensor
            input_budget: Token budget for the input
            target_budget: Token budget for the target

        Returns:
            Dictionary containing the masked image tensor, the input mask, the target mask, and the decoder attention mask
        """
        noise = torch.rand(num_tokens)
        ids_shuffle = torch.argsort(noise, dim=0)

        input_mask = torch.ones(num_tokens, dtype=torch.bool)
        input_mask[:input_budget] = 0
        input_mask = torch.gather(input_mask, dim=0, index=ids_shuffle)

        if target_budget is None:
            target_mask = ~input_mask
        else:
            target_mask = torch.ones(num_tokens, dtype=torch.bool)
            target_mask[input_budget:input_budget + target_budget] = 0
            target_mask = torch.gather(target_mask, dim=0, index=ids_shuffle)

        decoder_attention_mask = torch.zeros(num_tokens, dtype=torch.int)
        first_mask_token = torch.argmin(target_mask + torch.arange(target_mask.shape[0], device=target_mask.device) * 1e-6)
        decoder_attention_mask[first_mask_token] = (~target_mask).sum()  # Equiv. to target budget

        return {"tensor": tensor, "input_mask": input_mask, "target_mask": target_mask, "decoder_attention_mask": decoder_attention_mask}

    def sequence_token_mask(self, sequence_ids: str, max_tokens: int, input_budget: int, target_budget: int, keep_scheme: str, vocab_offset: int):
        """Applies input and target masking to a sequence of tokens (e.g. DINOv2 global tokens)
        The keep probability is sampled from a cosine schedule and does not depend on the number of tokens in the sequence.
        If the keep probability results in a sequence that is too long, then it is lowered until the sequence is short enough.

        Args:
            sequence_ids: Sequence ids
            max_tokens: Maximum number of tokens in the sequence
            input_budget: Token budget for the input
            target_budget: Token budget for the target
            keep_scheme: Scheme for sampling the keep probability
            vocab_offset: Offset to avoid overlap with sentinel tokens

        Returns:
            Dictionary containing the masked sequence tensor, the input mask, the target mask, and the decoder attention mask
        """
        seq_ids = sequence_ids 
        seq_ids = seq_ids + vocab_offset # Avoid overlap with sentinel tokens (needs to be substracted after decoding)

        # If input budget is 0, treat it as if the whole sequence is completely masked
        if input_budget == 0:
            keep_prob = 0.
            input_seq_ids = []
            _, target_seq_ids = simple_span_masking(seq_ids, self.sentinel_to_id, keep_prob)
        else:
            if keep_scheme == 'random':
                keep_prob = sample_uniform(0, 1)
            elif keep_scheme == 'all':
                keep_prob = 1.0
            elif keep_scheme == 'binary':
                keep_prob = random.choice([0., 1.])
            else:
                raise ValueError(f"Invalid keep scheme for sequence masking: {keep_scheme}")

            input_seq_ids, target_seq_ids = simple_span_masking(seq_ids, self.sentinel_to_id, keep_prob)
            # Keep lowering the keep_prob while we are over-budget
            while len(input_seq_ids) > input_budget:
                keep_prob = keep_prob * self.keep_prob_decay_factor
                input_seq_ids, target_seq_ids = simple_span_masking(seq_ids, self.sentinel_to_id, keep_prob)

        # Span masking can add up to (max_tokens + 1) * 2 tokens for input + target
        max_length = (max_tokens + 1) * 2
        tensor = torch.ones(max_length, dtype=torch.int) * self.pad_id
        input_mask = torch.ones(max_length, dtype=torch.bool)
        target_mask = torch.ones(max_length, dtype=torch.bool)
        decoder_attention_mask = torch.zeros(max_length, dtype=torch.int)

        # Set input and input mask
        tensor[:len(input_seq_ids)] = torch.tensor(input_seq_ids, dtype=torch.int)
        input_mask[:len(input_seq_ids)] = 0

        if target_budget is None or len(target_seq_ids) <= target_budget:
            tensor[input_budget:input_budget + len(target_seq_ids)] = torch.tensor(target_seq_ids, dtype=torch.int)
            target_mask[input_budget:input_budget + len(target_seq_ids)] = 0
            decoder_attention_mask[input_budget:input_budget + len(target_seq_ids)] = 1
        else:
            # Randomly choose sentinel token.
            sentinel_indices = [i for i, token_id in enumerate(target_seq_ids) if token_id in self.sentinel_ids]
            # If there is more than 1 sentinel, avoid sampling the very last one which indicates the end of the sequence
            chosen_sentinel = np.random.randint(max(1, len(sentinel_indices) - 1))
            # If length starting at this token g.t. budget, truncate until budget is reached
            if len(target_seq_ids) - sentinel_indices[chosen_sentinel] >= target_budget:
                target_seq_ids = target_seq_ids[sentinel_indices[chosen_sentinel]:sentinel_indices[chosen_sentinel] + target_budget]
            # Otherwise, select earliest sentinel token such that we don't go over budget
            # Note: We could also use the randomly chosen sentinel token, but that would waste budget
            else:
                for idx in sentinel_indices:
                    if len(target_seq_ids) - idx <= target_budget:
                        target_seq_ids = target_seq_ids[idx:]
                        break

            tensor[input_budget:input_budget + len(target_seq_ids)] = torch.tensor(target_seq_ids, dtype=torch.int)
            target_mask[input_budget:input_budget + len(target_seq_ids)] = 0
            decoder_attention_mask[input_budget:input_budget + len(target_seq_ids)] = 1

        return {"tensor": tensor, "input_mask": input_mask, "target_mask": target_mask, "decoder_attention_mask": decoder_attention_mask}

    def sequence_mask(self, sequence: Union[str, List[str]], max_tokens: int, input_budget: int, target_budget: int, keep_scheme: str):
        """Applies input and target masking to a sequence

        The keep probability is sampled from a cosine schedule and does not depend on the number of tokens in the sequence.
        If the keep probability results in a sequence that is too long, then it is lowered until the sequence is short enough.

        Args:
            sequence: Sequence, can be either a str or list of strings
            max_tokens: Maximum number of tokens in the sequence
            input_budget: Token budget for the input
            target_budget: Token budget for the target
            keep_scheme: Scheme for sampling the keep probability

        Returns:
            Dictionary containing the masked sequence tensor, the input mask, the target mask, and the decoder attention mask
        """
        if isinstance(sequence, str):
            # Tokenize the sequence and get the ids
            seq_ids: List[int] = self.text_tokenizer.encode(sequence).ids
            # Add EOS to all sequences
            seq_ids.append(self.eos_id)
            # Truncate sequence
            seq_ids = seq_ids[:max_tokens]

            # Use default span masking
            span_masking_fn = simple_span_masking

        elif isinstance(sequence, list):
            # Tokenize the sequence chunks and get the ids
            encoded_seq_chunks = self.text_tokenizer.encode_batch(sequence)
            seq_ids: List[List[int]] = [seq.ids for seq in encoded_seq_chunks]
            # Add EOS as an extra chunk
            seq_ids.append([self.eos_id])
            # Truncate sequence to keep all chunks below max token length
            cumulative_token_count = np.cumsum(np.array([len(chunk) for chunk in seq_ids]))
            seq_ids = [chunk for (chunk, token_count) in zip(seq_ids, cumulative_token_count) if token_count <= max_tokens]

            # Span mask over chunks
            span_masking_fn = chunk_span_masking

        else:
            raise ValueError(f"Invalid sequence: {sequence}")


        # If input budget is 0, treat it as if the whole sequence is completely masked
        if input_budget == 0:
            keep_prob = 0.
            input_seq_ids = []
            _, target_seq_ids = span_masking_fn(seq_ids, self.sentinel_to_id, keep_prob)
        else:
            if keep_scheme == 'random':
                keep_prob = sample_uniform(0, 1)
            elif keep_scheme == 'all':
                keep_prob = 1.0
            elif keep_scheme == 'binary':
                keep_prob = random.choice([0., 1.])
            else:
                raise ValueError(f"Invalid keep scheme for sequence masking: {keep_scheme}")

            input_seq_ids, target_seq_ids = span_masking_fn(seq_ids, self.sentinel_to_id, keep_prob)
            # Keep lowering the keep_prob while we are over-budget
            while len(input_seq_ids) > input_budget:
                keep_prob = keep_prob * self.keep_prob_decay_factor
                input_seq_ids, target_seq_ids = span_masking_fn(seq_ids, self.sentinel_to_id, keep_prob)

        # Span masking can add up to (max_tokens + 1) * 2 tokens for input + target
        max_length = (max_tokens + 1) * 2
        tensor = torch.ones(max_length, dtype=torch.int) * self.pad_id
        input_mask = torch.ones(max_length, dtype=torch.bool)
        target_mask = torch.ones(max_length, dtype=torch.bool)
        decoder_attention_mask = torch.zeros(max_length, dtype=torch.int)

        # Set input and input mask
        tensor[:len(input_seq_ids)] = torch.tensor(input_seq_ids, dtype=torch.int)
        input_mask[:len(input_seq_ids)] = 0

        if target_budget is None or len(target_seq_ids) <= target_budget:
            tensor[input_budget:input_budget + len(target_seq_ids)] = torch.tensor(target_seq_ids, dtype=torch.int)
            target_mask[input_budget:input_budget + len(target_seq_ids)] = 0
            decoder_attention_mask[input_budget:input_budget + len(target_seq_ids)] = 1
        else:
            # Randomly choose sentinel token.
            sentinel_indices = [i for i, token_id in enumerate(target_seq_ids) if token_id in self.sentinel_ids]
            # If there is more than 1 sentinel, avoid sampling the very last one which indicates the end of the sequence
            chosen_sentinel = np.random.randint(max(1, len(sentinel_indices) - 1))
            # If length starting at this token g.t. budget, truncate until budget is reached
            if len(target_seq_ids) - sentinel_indices[chosen_sentinel] >= target_budget:
                target_seq_ids = target_seq_ids[sentinel_indices[chosen_sentinel]:sentinel_indices[chosen_sentinel] + target_budget]
            # Otherwise, select earliest sentinel token such that we don't go over budget
            # Note: We could also use the randomly chosen sentinel token, but that would waste budget
            else:
                for idx in sentinel_indices:
                    if len(target_seq_ids) - idx <= target_budget:
                        target_seq_ids = target_seq_ids[idx:]
                        break

            tensor[input_budget:input_budget + len(target_seq_ids)] = torch.tensor(target_seq_ids, dtype=torch.int)
            target_mask[input_budget:input_budget + len(target_seq_ids)] = 0
            decoder_attention_mask[input_budget:input_budget + len(target_seq_ids)] = 1

        return {"tensor": tensor, "input_mask": input_mask, "target_mask": target_mask, "decoder_attention_mask": decoder_attention_mask}


    def sequence_emb_mask_span(self, emb_tensor: torch.Tensor, max_tokens: int, input_budget: int, target_budget: int, keep_scheme: str):
        """Applies input masking to an sequence embedding tensor, target masking is not supported with sequence embeddings

        Args:
            emb_tensor: Sequence embedding tensor
            max_tokens: Maximum number of tokens in the sequence
            input_budget: Token budget for the input
            target_budget: Token budget for the target (unused for now)
            keep_scheme: Scheme for sampling the keep probability

        Returns:
            Dictionary containing the masked sequence embedding tensor, the input mask, the target mask, and the decoder attention mask
        """
        # Only supported as input modality now

        # Make fake seq ids for sequence embeddings to reuse simple_span_masking function
        fake_seq_ids = []
        emb_dict = {}
        id_num = len(self.sentinel_ids)
        emb_ind = 0
        while(len(fake_seq_ids) < len(emb_tensor)):
            if id_num not in self.sentinel_ids: # replace with T5 sentinel_id
                fake_seq_ids.append(id_num)
                emb_dict[id_num] = emb_tensor[emb_ind, :]
                emb_ind += 1
            id_num += 1
                
        # Truncate sequence
        fake_seq_ids = fake_seq_ids[:max_tokens]

        # If input budget is 0, treat it as if the whole sequence is completely masked
        if input_budget == 0:
            keep_prob = 0.
            fake_input_seq_ids = []
            _, fake_target_seq_ids = simple_span_masking(fake_seq_ids, self.sentinel_to_id, keep_prob)
        else:
            if keep_scheme == 'random':
                keep_prob = sample_uniform(0, 1)
            elif keep_scheme == 'all':
                keep_prob = 1.0
            elif keep_scheme == 'binary':
                keep_prob = random.choice([0., 1.])
            else:
                raise ValueError(f"Invalid keep scheme for sequence masking: {keep_scheme}")

            fake_input_seq_ids, fake_target_seq_ids = simple_span_masking(fake_seq_ids, self.sentinel_to_id, keep_prob)
            # Keep lowering the keep_prob while we are over-budget
            while len(fake_input_seq_ids) > input_budget:
                keep_prob = keep_prob * self.keep_prob_decay_factor
                fake_input_seq_ids, fake_target_seq_ids = simple_span_masking(fake_seq_ids, self.sentinel_to_id, keep_prob)

        # Span masking can add up to max_tokens tokens for input
        max_length = max_tokens
        tensor = torch.zeros((max_length, emb_tensor.shape[1]), dtype=torch.float32)
        input_mask = torch.ones(max_length, dtype=torch.bool)
        target_mask = torch.ones(max_length, dtype=torch.bool)
        decoder_attention_mask = torch.zeros(max_length, dtype=torch.int)

        # Put tensor values back based on the fake seq ids
        for i_, fake_id in enumerate(fake_input_seq_ids):
            if fake_id in self.sentinel_ids:
                tensor[i_, :] = torch.zeros_like(emb_tensor[0,:]) # TODO replace to learned embeddings later
            else:
                tensor[i_, :] = emb_dict[fake_id]
            
        # Set input and input mask
        input_mask[:len(fake_input_seq_ids)] = 0

        return {"tensor": tensor, "input_mask": input_mask, "target_mask": target_mask, "decoder_attention_mask": decoder_attention_mask}
    
    
    def __call__(self, mod_dict):
        """Applies input and target masking to a dictionary of modalities

        Args:
            mod_dict: Dictionary of modalities

        Returns:
            Dictionary containing the masked modalities
        """
        if self.sampling_weights is not None:
            # Sample masking scheme according to a list of weights
            dir_idx = torch.multinomial(self.sampling_weights, 1).item()
        else:
            # Randomly sample masking scheme
            dir_idx = random.randint(0, self.num_dirichlets - 1)

        num_input_tokens = random.randint(*self.input_tokens_range)
        num_target_tokens = random.randint(*self.target_tokens_range) if self.target_tokens_range is not None else None

        input_token_budget = self.input_token_budget(num_input_tokens, dir_idx)

        if num_target_tokens is not None:
            target_token_budget = self.target_token_budget(input_token_budget, num_target_tokens, dir_idx)
        else:
            target_token_budget = [None] * self.num_modalities

        masked_mod_dict = {}
        for (mod_name, mod_info), input_budget, target_budget in zip(self.modality_info.items(), input_token_budget, target_token_budget):
            mod_type = mod_info['type']
            mod_name_load = mod_name if mod_name in mod_dict else get_transform_key(mod_name)
            if mod_type == 'img':
                masked_mod_dict[mod_name] = self.image_mask(mod_dict[mod_name_load], mod_info['max_tokens'], input_budget, target_budget)
            elif mod_type == 'seq':
                keep_scheme = 'random' if ('keep' not in mod_info) else mod_info['keep'][dir_idx]
                masked_mod_dict[mod_name] = self.sequence_mask(mod_dict[mod_name_load], mod_info['max_tokens'], input_budget, target_budget, keep_scheme)
            elif mod_type == 'seq_token':
                keep_scheme = 'random' if ('keep' not in mod_info) else mod_info['keep'][dir_idx]
                vocab_offset =  mod_info.get('vocab_offset', 0) # Check if any space is allocated to sentinel tokens and other special tokens
                masked_mod_dict[mod_name] = self.sequence_token_mask(mod_dict[mod_name_load], mod_info['max_tokens'], input_budget, target_budget, keep_scheme, vocab_offset=vocab_offset)
            elif mod_type == "seq_emb":
                keep_scheme = 'random' if ('keep' not in mod_info) else mod_info['keep'][dir_idx]
                masked_mod_dict[mod_name] = self.sequence_emb_mask_span(mod_dict[mod_name_load], mod_info['max_tokens'], input_budget, target_budget, keep_scheme)
            else:
                raise ValueError(f"Invalid modality type: {mod_type}")

        return masked_mod_dict


class TransferMasking(object):
    def __init__(self,
                 modality_info: Dict,
                 text_tokenizer: Optional[Tokenizer],
                 input_modalities: List[str],
                 target_modalities: List[str]):
        """Performs masking for transfer on a dict of modalities (both image based and sequence based modalities),
        by specifying which modalities are inputs and which are targets.

        Args:
            modality_info: Dict with the modalities and their corresponding information
            text_tokenizer: Tokenizer to use for text modalities
            input_modalities: List of modalities to use as input
            target_modalities: List of modalities to use as target
        """
        self.modality_info = modality_info
        self.num_modalities = len(modality_info)
        self.min_tokens = torch.tensor([mod['min_tokens'] for mod in modality_info.values()])
        self.max_tokens = torch.tensor([mod['max_tokens'] for mod in modality_info.values()])
        self.mod_is_img = torch.tensor([mod['type'] == 'img' for mod in modality_info.values()])

        self.input_modalities = set(input_modalities)
        self.target_modalities = set(target_modalities)

        # Tokenizer for text modalities
        self.text_tokenizer = text_tokenizer
        if self.text_tokenizer is not None:
            self.keep_prob_decay_factor = 0.9
            self.sentinel_to_id = get_sentinel_to_id_mapping(text_tokenizer)
            self.sentinel_ids = set(self.sentinel_to_id.values())
            self.pad_id = text_tokenizer.token_to_id("[PAD]")
            self.eos_id = text_tokenizer.token_to_id("[EOS]")


    def input_image(self, tensor: torch.Tensor, num_tokens: int):
        """Applies masking for an image given as input

        Args:
            tensor: Image tensor
            num_tokens: Number of tokens in the tensor

        Returns:
            Dictionary containing the masked image tensor, the input mask, the target mask, and the decoder attention mask
        """

        # Input mask
        input_mask = torch.zeros(num_tokens, dtype=torch.bool)
        # Target mask
        target_mask = torch.ones(num_tokens, dtype=torch.bool)
        # Decoder attention mask
        decoder_attention_mask = torch.zeros(num_tokens, dtype=torch.int)

        return {"tensor": tensor, "input_mask": input_mask, "target_mask": target_mask, "decoder_attention_mask": decoder_attention_mask}

    def target_image(self, tensor: torch.Tensor, num_tokens: int):
        """Applies masking for an image given as target

        Args:
            tensor: Image tensor
            num_tokens: Number of tokens in the tensor

        Returns:
            Dictionary containing the masked image tensor, the input mask, the target mask, and the decoder attention mask
        """

        # Input mask
        input_mask = torch.ones(num_tokens, dtype=torch.bool)
        # Target mask
        target_mask = torch.zeros(num_tokens, dtype=torch.bool)
        # Decoder attention mask
        decoder_attention_mask = torch.zeros(num_tokens, dtype=torch.int)
        decoder_attention_mask[0] = num_tokens

        return {"tensor": tensor, "input_mask": input_mask, "target_mask": target_mask, "decoder_attention_mask": decoder_attention_mask}


    def input_sequence(self, sequence_str: str, max_tokens: int):
        """Applies masking for a sequence given as input

        Args:
            sequence_str: Sequence string
            max_tokens: Maximum number of tokens in the sequence

        Returns:
            Dictionary containing the masked sequence string, the input mask, the target mask, and the decoder attention mask
        """
        # Tokenize the text and get the ids
        seq_ids = self.text_tokenizer.encode(sequence_str).ids
        # Add EOS to all sequences
        seq_ids.append(self.eos_id)
        # Truncate sequence
        seq_ids = seq_ids[:max_tokens]

        keep_prob = 1.
        input_seq_ids, target_seq_ids = simple_span_masking(seq_ids, self.sentinel_to_id, keep_prob)

        # Span masking can add up to (max_tokens + 1) * 2 tokens for input + target
        max_length = (max_tokens + 1) * 2
        tensor = torch.ones(max_length, dtype=torch.int) * self.pad_id
        input_mask = torch.ones(max_length, dtype=torch.bool)
        target_mask = torch.ones(max_length, dtype=torch.bool)
        decoder_attention_mask = torch.zeros(max_length, dtype=torch.int)

        # Set input and input mask
        tensor[:len(input_seq_ids)] = torch.tensor(input_seq_ids, dtype=torch.int)
        input_mask[:len(input_seq_ids)] = 0

        tensor[max_tokens:max_tokens + len(target_seq_ids)] = torch.tensor(target_seq_ids, dtype=torch.int)
        target_mask[max_tokens:max_tokens + len(target_seq_ids)] = 0
        decoder_attention_mask[max_tokens:max_tokens + len(target_seq_ids)] = 1


        return {"tensor": tensor, "input_mask": input_mask, "target_mask": target_mask, "decoder_attention_mask": decoder_attention_mask}


    def target_sequence(self, sequence_str: str, max_tokens: int):
        """Applies masking for a sequence given as target

        Args:
            sequence_str: Sequence string
            max_tokens: Maximum number of tokens in the sequence

        Returns:
            Dictionary containing the masked sequence string, the input mask, the target mask, and the decoder attention mask
        """
        # Tokenize the text and get the ids
        seq_ids = self.text_tokenizer.encode(sequence_str).ids
        # Add EOS to all sequences
        seq_ids.append(self.eos_id)
        # Truncate sequence
        seq_ids = seq_ids[:max_tokens]

        keep_prob = 0.
        input_seq_ids = []
        _, target_seq_ids = simple_span_masking(seq_ids, self.sentinel_to_id, keep_prob)

        # Span masking can add up to (max_tokens + 1) * 2 tokens for input + target
        max_length = (max_tokens + 1) * 2
        tensor = torch.ones(max_length, dtype=torch.int) * self.pad_id
        input_mask = torch.ones(max_length, dtype=torch.bool)
        target_mask = torch.ones(max_length, dtype=torch.bool)
        decoder_attention_mask = torch.zeros(max_length, dtype=torch.int)

        # Set input and input mask
        tensor[:len(input_seq_ids)] = torch.tensor(input_seq_ids, dtype=torch.int)
        input_mask[:len(input_seq_ids)] = 0

        tensor[max_tokens:max_tokens + len(target_seq_ids)] = torch.tensor(target_seq_ids, dtype=torch.int)
        target_mask[max_tokens:max_tokens + len(target_seq_ids)] = 0
        decoder_attention_mask[max_tokens:max_tokens + len(target_seq_ids)] = 1

        return {"tensor": tensor, "input_mask": input_mask, "target_mask": target_mask,
                "decoder_attention_mask": decoder_attention_mask}

    def __call__(self, mod_dict):
        """Applies input and target masking to a dictionary of modalities

        Args:
            mod_dict: Dictionary of modalities

        Returns:
            Dictionary containing the masked modalities
        """
        masked_mod_dict = {}
        for mod_name, mod_info in self.modality_info.items():
            mod_type = mod_info['type']
            if mod_type == 'img' and mod_name in self.input_modalities:
                masked_mod_dict[mod_name] = self.input_image(mod_dict[mod_name], mod_info['max_tokens'])
            elif mod_type == 'img' and mod_name in self.target_modalities:
                masked_mod_dict[mod_name] = self.target_image(mod_dict[mod_name], mod_info['max_tokens'])
            elif mod_type == 'seq' and mod_name in self.input_modalities:
                masked_mod_dict[mod_name] = self.input_sequence(mod_dict[mod_name], mod_info['max_tokens'])
            elif mod_type == 'seq' and mod_name in self.target_modalities:
                masked_mod_dict[mod_name] = self.target_sequence(mod_dict[mod_name], mod_info['max_tokens'])
            else:
                raise ValueError(f"Invalid modality type: {mod_type} or modality name not in input or target modalities: {mod_name}")

        if 'mask_valid' in mod_dict:
            masked_mod_dict['mask_valid'] = mod_dict['mask_valid']

        return masked_mod_dict