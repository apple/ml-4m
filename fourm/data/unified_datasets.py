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
import copy
import io
import itertools
import os
import re
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional

import braceexpand
import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import IterableDataset
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from webdataset.filters import pipelinefilter, reraise_exception
from webdataset.handlers import warn_and_continue

try:
    # Optionally load huggingface datasets
    from datasets import load_dataset
    from datasets.distributed import split_dataset_by_node
except ImportError:
    print("Huggingface datasets not installed. Please install with `pip install datasets`.")

from fourm.data.masking import TransferMasking, UnifiedMasking
from fourm.data.modality_transforms import (CropSettingsTransform, IdentityTransform,
                                      MaskTransform, UnifiedDataTransform,
                                      get_transform_key)
from fourm.data.multimodal_dataset_folder import MultiModalDatasetFolder
from fourm.utils.dist import get_rank, get_world_size


def build_fm_pretraining_dataset(
        data_path, all_domains, modality_info, modality_transforms, 
        image_augmenter, text_tokenizer, 
        input_tokens_range, target_tokens_range,
        sampling_weights=None):
    """Builds the FourM pre-training dataset based on the given arguments.
    This function should mainly used for smaller datasets (e.g. validation sets), 
    while large training sets should be loaded with build_wds_fm_pretraining_dataloader in webdataset format.
    
    Args:
        data_path: Path to the dataset.
        all_domains: List of all modalities to be used.
        modality_info: Dictionary containing information about the modalities.
        modality_transforms: Dictionary containing the transforms for each modality.
        image_augmenter: Image augmenter.
        text_tokenizer: Text tokenizer (for sequence modalities).
        input_tokens_range: Range of the input token budget.
        target_tokens_range: Range of the target token budget.
        sampling_weights: Sampling weights for the mixture of Dirichlet distributions.

    Returns:
        FourM pre-training dataset as a PyTorch Dataset.
    """

    transform = transforms.Compose([
        UnifiedDataTransform(transforms_dict=modality_transforms, image_augmenter=image_augmenter),
        UnifiedMasking(modality_info=modality_info, text_tokenizer=text_tokenizer,
                       input_tokens_range=input_tokens_range, target_tokens_range=target_tokens_range,
                       sampling_weights=sampling_weights),
         ])

    # Remove vq domains that require a tokenizer
    modalities_without_vq = [mod for mod in all_domains if not modality_info[mod].get("requires_tokenizer", False)]
    # If we are using a pre-tokenized modality, we default to pre-computed crop settings
    if any([modality_info[domain].get("pretokenized", False) for domain in all_domains]):
        modalities_without_vq.append("crop_settings")
        modality_transforms = copy.deepcopy(modality_transforms)
        modality_transforms["crop_settings"] = CropSettingsTransform()

    modality_paths = {mod: modality_info[mod]['path'] for mod in modality_info if modality_info[mod].get('path', None) is not None}
    
    return MultiModalDatasetFolder(root=data_path, modalities=modalities_without_vq, modality_paths=modality_paths,
                                   modality_transforms=modality_transforms, transform=transform)


def build_fm_transfer_dataset(
    data_path, modality_info, transform, modality_transforms, all_domains, 
    load_mask_valid: bool = False, max_samples: Optional[int] = None, 
    pre_shuffle: bool = False, cache: bool = False):
    """Builds the FourM transfer dataset based on the given arguments.
    
    Args:
        data_path: Path to the dataset.
        modality_info: Dictionary containing information about the modalities.
        transform: Transform to be applied to the dataset.
        modality_transforms: Dictionary containing the transforms for each modality.
        all_domains: List of all modalities to be used.
        load_mask_valid: Whether to load the mask_valid "modality".
        max_samples: Maximum number of samples to load.
        pre_shuffle: Whether to shuffle the dataset before loading.
        cache: Whether to cache the dataset in memory.

    Returns:
        FourM transfer dataset as a PyTorch Dataset.
    """

    # Remove vq domains that require a tokenizer
    modalities_without_vq = [mod for mod in all_domains if not modality_info[mod].get("requires_tokenizer", False)]
    # If we are using a pre-tokenized modality, we default to pre-computed crop settings
    if any([modality_info[domain].get("pretokenized", False) for domain in all_domains]):
        modalities_without_vq.append("crop_settings")
        modality_transforms = copy.deepcopy(modality_transforms)
        modality_transforms["crop_settings"] = CropSettingsTransform()

    if load_mask_valid:
        modalities_without_vq.append("mask_valid")
        modality_transforms = copy.deepcopy(modality_transforms)
        modality_transforms["mask_valid"] = MaskTransform()

    modality_paths = {mod: modality_info[mod]['path'] for mod in modality_info if modality_info[mod].get('path', None) is not None}

    return MultiModalDatasetFolder(root=data_path, modalities=modalities_without_vq, modality_paths=modality_paths,
                                   modality_transforms=modality_transforms, transform=transform, max_samples=max_samples, 
                                   pre_shuffle=pre_shuffle, cache=cache)


### Webdatasets (wds) functions

def _keyless_map(data, f, handler=reraise_exception):
    """Map samples without adding __key__."""
    for sample in data:
        try:
            result = f(sample)
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
        if result is None:
            continue
        yield result

map = pipelinefilter(_keyless_map)

def check_dots(s):
    if '.gz' in s:
        return s.count('.') == 2
    return s.count('.') == 1

def remove_ext_with_gz(s):
    if s.endswith('.gz'):
        s = s.replace(".gz", "")
    return os.path.splitext(s)[0]

def wds_decoder(key, value):
    if key == "png" or key.endswith(".png"):
        img = Image.open(io.BytesIO(value))
        return img
    elif key == "jpg" or key.endswith(".jpg"):
        img = Image.open(io.BytesIO(value))
        return img
    elif key == "jpeg" or key.endswith(".jpeg"):
        img = Image.open(io.BytesIO(value))
        return img
    elif key == 'npy' or key.endswith("npy"):
        content = np.load(io.BytesIO(value), allow_pickle=True)
        # try:
        #     content = np.load(io.BytesIO(value))
        # except:
        #     content = np.load(io.BytesIO(value), allow_pickle=True)
        return content
    elif key == "jpx" or key.endswith('.jpx'):
        img = Image.open(io.BytesIO(value))
        return img
    elif 'output' in key:
        return int(value)
    else:
        # If not an image, use the basic handlers (.txt, .json, .pickle, .npz, ...)
        # See https://github.com/webdataset/webdataset/blob/main/webdataset/autodecode.py
        return None

def repeat_fn(src, n_repeats=5):
    """
    Repeat each sample n_repeats times.
    E.g. A B C ... repeated 3 times becomes A A A B B B C C C ...
    Depending on the downstream application, a shuffle should be added after this.
    """
    for sample in src:
        for _ in range(n_repeats):
            yield sample
            
def remove_extensions(sample):
    """
    In webdatasets, we identify the type of a given modality by adding an extension
    in the form f"{modality_name}.{modality_extension}", e.g. "rgb.jpg" or "caption.json".
    This function removes them and returns a dictionary of {f"{modality_name}": modality}.
    """
    return {remove_ext_with_gz(k): v for k, v in sample.items()}

def filter_metadata(sample, metadata=['__key__', '__url__', 'file_name', 'class_name', 'class_idx']):
    """ Filters out non-modality entries specified in metadata when loading tar files with webdatasets. """
    return {k: v for k, v in sample.items() if k not in metadata}

def apply_modality_transforms(sample, modality_transforms):
    """ Applies a dictionary of modality-specific transforms to a dictionary of modalities. """
    return {k: (modality_transforms[get_transform_key(k)](v) if k in modality_transforms else v) for k, v in sample.items() }

def tok_to_int64(sample):
    """
    Pre-computed tokens are saved as int16, but we need them as int64 instead.
    """
    return {k: (v.astype('int64') if 'tok_' in k else v) for k, v in sample.items()}

def rename_modalities(sample, modality_paths):
    """
    Renames modalities to their corresponding names in modality_paths.
    """
    return {out_path: sample[loaded_path] for out_path, loaded_path in modality_paths.items()}

def extract_modality_names(s):
    # Regular expression pattern to match anything enclosed in '{' and '}', and comma separated
    pattern = r'\{([^}]*)\}'
    match = re.search(pattern, s)
    return match.group(1).split(',') if match else []

def identity(sample):
    """ Identity function that does nothing. """
    return sample

def multi_tarfile_samples(src_iter: Iterable[Dict[str, Any]], 
                          modality_name_map: Dict[str, str] = None, 
                          handler: Callable[[Exception], bool] = warn_and_continue):
    """Webdataset does not support splitting up shards by modality, so we need to do this manually.
    Usually, we would need to save all modalities in the same tar file, e.g. shard_root_train/{00000..12345}.tar, 
    where each shard contains 1000 samples and each sample contains all modalities.
    This is not flexible when adding new modalities, so we instead save each modality in a separate tar file,
    e.g. shard_root_train_rgb/{00000..12345}.tar, shard_root_train_caption/{00000..12345}.tar, etc., where each shard contains
    again 1000 samples, but each sample contains only one modality. All samples in all shards have to be aligned.

    This function takes an iterator over shard URLs, where we use brace expansion to specify multiple tar files per modality.
    E.g. shard_root_train_[rgb,caption]/00123.tar will be expanded to shard_root_train_rgb/00123.tar and shard_root_train_caption/00123.tar,
    and the samples from these two tar files will be combined into a single sample.

    Args:
        src_iter: Iterator over shards that *already brace expanded the shard numbers*, 
            e.g. {'url': 'shard_root_train_[rgb,caption]/00000.tar'}, {'url': 'shard_root_train_[rgb,caption]/00001.tar'}, ...
            This function will also work when no square braces for multiple modalities are used, e.g. {'url': 'shard_root_train/00000.tar'}, ...
            It can be a drop-in replacement for wds.tarfile_samples.
        modality_name_map: Optional dictionary specifying a mapping from modality folder names to arbitrary other names.
        handler: Function that handles exceptions. If it returns True, the shard is skipped. If it returns False, the function exits.

    Yields:
        Dictionary of aligned samples from all modalities.
    """
    for src in src_iter:
        
        # Multi tar file URLs use brace expansion with square braces
        multi_tar_urls = src['url'].translate(str.maketrans('[]', '{}'))
        modality_names = extract_modality_names(multi_tar_urls)
        if len(modality_names) == 0:
            # Case where multi-modal braceexpand is not used, e.g. shard_dir/shard00000.tar
            modality_names = [None]
            multi_tar_urls = [multi_tar_urls]
        elif len(modality_names) == 1:
            # Brace expand doesn't work with a single entry, e.g. shard_dir/[foo]/shard00000.tar
            multi_tar_urls = [multi_tar_urls.replace("{", "").replace("}", "")]
        else:
            # Remaining cases where multiple modalities are specified, e.g. shard_dir/[foo,bar]/shard00000.tar
            multi_tar_urls = list(braceexpand.braceexpand(multi_tar_urls))

        # Create tar iterators for shards of all modalities
        tar_iters = [wds.tarfile_samples([{'url': tar_url}]) for tar_url in multi_tar_urls]
        
        try:
            # Loop over these iterators in parallel and combine the tar files from different modalities
            for multi_tar_files in zip(*tar_iters):
                
                merged_dict = {}
                merged_dict['__key__'] = multi_tar_files[0]['__key__']
                merged_dict['__url__'] = src['url']
                
                for modality_name, modality_dict in zip(modality_names, multi_tar_files):
                    _key = modality_dict.pop('__key__')
                    _url = modality_dict.pop('__url__')

                    if _key != merged_dict['__key__']:
                        raise ValueError(f"Divergence detected! Trying to merge keys {_key} of {modality_name} and {merged_dict['__key__']} of merged_dict with modalities {merged_dict.keys()}.")
                        
                    tar_is_multimodal = len(modality_dict) > 1
                    for k, v in modality_dict.items():
                        if tar_is_multimodal or check_dots(k) or modality_name is None:
                            # We don't change the keys in the following cases:
                            # 1. The shard contains multiple modalities. Then they *have* to follow the idx.modality_id.ext convention
                            # 2. If any key contains a dot, this means it already has the idx.modality_id.ext format (idx. is already removed at this stage)
                            # 3. If the modality name is None, no modality folder was specified (see beginning of function)
                            merged_dict[k] = v
                        else:
                            mapped_name = modality_name if modality_name_map is None else modality_name_map.get(modality_name, modality_name)
                            merged_dict[f'{mapped_name}.{k}'] = v

                yield merged_dict

        except Exception as e:
            print(e)
            print(f"Exception occurred while processing {src['url']}.")
            if handler(e):
                print('Skipping shard...')
                continue
            else:
                break

def build_wds_fm_pretraining_dataloader(
        data_path, all_domains, modality_info, modality_transforms, image_augmenter, 
        text_tokenizer, input_tokens_range, target_tokens_range,
        num_gpus, num_workers, batch_size, epoch_size, sampling_weights=None, modality_name_map=None,
        shuffle_buffer_load=1000, shuffle_buffer_repeat=5000, n_repeats=5):
    """Builds the WebDataset FourM pre-training dataloader based on the given arguments.
    
    Args:
        data_path: Path to the dataset.
        all_domains: List of all modalities to be used.
        modality_info: Dictionary containing information about the modalities.
        modality_transforms: Dictionary containing the transforms for each modality.
        image_augmenter: Image augmenter.
        text_tokenizer: Text tokenizer (for sequence modalities).
        input_tokens_range: Range of the input token budget.
        target_tokens_range: Range of the target token budget.
        num_gpus: Number of GPUs.
        num_workers: Number of workers.
        batch_size: Batch size.
        epoch_size: Number of samples per "epoch". (Here, epoch refers to an interrupted training loop without evaluation or checkpointing).
        sampling_weights: Sampling weights for the mixture of Dirichlet distributions.
        modality_name_map: Optional dictionary specifying a mapping from modality folder names to arbitrary other names.
        shuffle_buffer_load: Shuffle buffer size when loading samples from tar files (first shuffle).
        shuffle_buffer_repeat: Shuffle buffer size after repeating samples (second shuffle).
        n_repeats: Number of times to repeat each sample.

    Returns:
        FourM pre-training dataloader as a WebDataset DataLoader.
    """

    modality_paths = {mod: modality_info[mod].get('path', None) or mod for mod in modality_info}

    # Remove vq domains that require a tokenizer
    modalities_without_vq = [mod for mod in all_domains if not modality_info[mod].get("requires_tokenizer", False)]
    # If we are using a pre-tokenized modality, we default to pre-computed crop settings
    if any([modality_info[domain].get("pretokenized", False) for domain in all_domains]):
        modalities_without_vq.append("crop_settings")
        modality_transforms = copy.deepcopy(modality_transforms)
        modality_transforms["crop_settings"] = CropSettingsTransform()
        modality_paths["crop_settings"] = "crop_settings"

    # Webdatasets always adds __key__ to the dictionary, so we add a transform that does nothing with it
    modality_transforms["__key__"] = IdentityTransform()

    transform = transforms.Compose([
        UnifiedDataTransform(transforms_dict=modality_transforms, image_augmenter=image_augmenter),
        UnifiedMasking(modality_info=modality_info, text_tokenizer=text_tokenizer,
                       input_tokens_range=input_tokens_range, target_tokens_range=target_tokens_range,
                       sampling_weights=sampling_weights)
    ])
    
    datapipe = wds.DataPipeline(
        # Infinitely sample shards from the shard list with replacement. Each worker is seeded independently.
        wds.ResampledShards(data_path),
        partial(multi_tarfile_samples, modality_name_map=modality_name_map), # Extract individual samples from single or multi-modal tar files
        wds.shuffle(shuffle_buffer_load), # Shuffle with a buffer of given size
        wds.decode(wds_decoder), # Decode from bytes to PIL images, numpy arrays, etc.
        wds.filters.compose(partial(repeat_fn, n_repeats=n_repeats)), # Repeats each sample n times -> A A A B B B C C C ...
        wds.shuffle(shuffle_buffer_repeat), # Shuffle again with a buffer of given size
        wds.map(remove_extensions), # Remove "file extensions" from dictionary keys
        map(filter_metadata), # Remove non-task keys
        map(tok_to_int64), # Convert pre-computed tokens to int64
        map(partial(rename_modalities, modality_paths=modality_paths)), # Rename modalities to their corresponding names in modality_paths
        map(transform), # Apply data augmentation and masking
        wds.batched(batch_size, collation_fn=default_collate, partial=False)
            if batch_size is not None else map(identity), # Batching
    )

    if epoch_size is not None:
        batch_size_iter = batch_size if batch_size is not None else 1
        datapipe = datapipe.with_epoch(epoch_size // (num_gpus * num_workers * batch_size_iter)) # Pre-define iterator length
    
    if batch_size is not None:
        # Perform multi-threaded dataloading
        return wds.WebLoader(datapipe, num_workers=num_workers, batch_size=None)
    else:
        return datapipe


def build_wds_divae_dataloader(
    data_path, modality_info, modality_transforms, image_augmenter,  
    num_gpus, num_workers, batch_size, epoch_size, shuffle_buffer_load=1000, 
    shuffle_buffer_repeat=5000, n_repeats=1):

    modality_paths = {mod: modality_info[mod].get('path', None) or mod for mod in modality_info}

    # Webdatasets always adds __key__ to the dictionary, so we add a transform that does nothing with it
    modality_transforms["__key__"] = IdentityTransform()

    transform = UnifiedDataTransform(transforms_dict=modality_transforms, image_augmenter=image_augmenter)

    datapipe = wds.DataPipeline(
        # Infinitely sample shards from the shard list with replacement. Each worker is seeded independently.
        wds.ResampledShards(data_path),
        multi_tarfile_samples, # Extract individual samples from single or multi-modal tar files
        wds.shuffle(shuffle_buffer_load), # Shuffle with a buffer of given size
        wds.decode(wds_decoder), # Decode from bytes to PIL images, numpy arrays, etc.
        wds.filters.compose(partial(repeat_fn, n_repeats=n_repeats)), # Repeats each sample n times -> A A A B B B C C C ...
        wds.shuffle(shuffle_buffer_repeat), # Shuffle again with a buffer of given size
        map(remove_extensions), # Remove "file extensions" from dictionary keys
        map(filter_metadata), # Remove non-task keys
        map(tok_to_int64), # Convert pre-computed tokens to int64
        map(partial(rename_modalities, modality_paths=modality_paths)), # Rename modalities to their corresponding names in modality_paths
        map(transform), # Apply data augmentation and masking
        wds.batched(batch_size, collation_fn=default_collate, partial=False)
            if batch_size is not None else map(identity), # Batching
    )

    if epoch_size is not None:
        batch_size_iter = batch_size if batch_size is not None else 1
        datapipe = datapipe.with_epoch(epoch_size // (num_gpus * num_workers * batch_size_iter)) # Pre-define iterator length
    
    if batch_size is not None:
        # Perform multi-threaded dataloading
        return wds.WebLoader(datapipe, num_workers=num_workers, batch_size=None)
    else:
        return datapipe


### Huggingface datasets functions

def text_to_caption(sample):
    """ Rename "text" to "caption". """
    return {'caption': sample['text']}


def build_huggingface_pretraining_dataloader(
        data_path, all_domains, modality_info, modality_transforms, image_augmenter, 
        text_tokenizer, input_tokens_range, target_tokens_range,
        num_gpus, num_workers, batch_size, epoch_size, split,
        streaming=True, rename_text_to_caption=True, shuffle_buffer_load=10_000, shuffle_seed=0):

    # Load huggingface dataset and split samples across workers. Shuffle samples in each worker
    dataset = load_dataset(data_path, split=split, streaming=streaming)
    dataset = split_dataset_by_node(dataset, rank=get_rank(), world_size=get_world_size())
    dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_load)

    modality_info = {mod: modality_info[mod] for mod in modality_info if mod in all_domains}
    
    transform = transforms.Compose([
        UnifiedDataTransform(transforms_dict=modality_transforms, image_augmenter=image_augmenter),
        UnifiedMasking(modality_info=modality_info, text_tokenizer=text_tokenizer,
                       input_tokens_range=input_tokens_range, target_tokens_range=target_tokens_range)
    ])
    
    datapipe = wds.DataPipeline(
        dataset,
        map(text_to_caption) if rename_text_to_caption else map(identity), # Rename "text" to "caption"
        map(filter_metadata), # Remove non-task keys
        map(transform), # Apply data augmentation and masking
        wds.batched(batch_size, collation_fn=default_collate, partial=False)
            if batch_size is not None else map(identity), # Batching
    )

    datapipe.n_shards = dataset.n_shards
    num_workers = min(num_workers, dataset.n_shards)

    if epoch_size is not None:
        batch_size_iter = batch_size if batch_size is not None else 1
        datapipe = datapipe.with_epoch(epoch_size // (num_gpus * num_workers * batch_size_iter)) # Pre-define iterator length

    if batch_size is not None:
        # Perform multi-threaded dataloading
        return wds.WebLoader(datapipe, num_workers=num_workers, batch_size=None)
    else:
        return datapipe


### Multi-dataset loading utils
def make_empty_mod_dict(modality_info):
    empty_mod_dicts = {}

    for mod_name, mod_info in modality_info.items():
        empty_mod = {}

        # Tensor
        if 'num_channels' in mod_info and 'input_size' in mod_info:
            # Handle image-like modalities
            max_tokens = mod_info['max_tokens']
            empty_mod['tensor'] = torch.zeros((mod_info['num_channels'], mod_info['input_size'], mod_info['input_size']), dtype=torch.float32)
        elif mod_name == 't5_caption':
            # Handle T5 embedding
            max_tokens = mod_info['max_tokens']
            orig_emb_dim = mod_info['encoder_embedding']().orig_emb_dim
            empty_mod['tensor'] = torch.zeros((max_tokens, orig_emb_dim), dtype=torch.float32)
        elif mod_info['type'] in ['seq', 'seq_emb', 'seq_token']:
            # Handle all other discrete sequence modalities
            max_tokens = (mod_info['max_tokens'] + 1) * 2
            empty_mod['tensor'] = torch.zeros((max_tokens), dtype=torch.int32)
        else:
            max_tokens = mod_info['max_tokens']
            empty_mod['tensor'] = torch.zeros((max_tokens), dtype=torch.int32)
            
        # Input and target masks
        empty_mod['input_mask'] = torch.ones((max_tokens), dtype=torch.bool)
        empty_mod['target_mask'] = torch.ones((max_tokens), dtype=torch.bool)

        # Decoder attention mask
        empty_mod['decoder_attention_mask'] = torch.zeros((max_tokens), dtype=torch.int32)
        
        empty_mod_dicts[mod_name] = empty_mod
        
    return empty_mod_dicts


class MixtureDataset(IterableDataset):
    def __init__(self, data_iters, weights, modality_info):
        self.orig_data_iters = data_iters
        self.data_iters = [iter(data_iter) for data_iter in data_iters]  # Create initial iterators
        self.sampling_probs = np.array(weights) / sum(weights)
        self.modality_info = modality_info

    def reset_iterator(self, idx):
        """ Reset the iterator when exhausted. """
        self.data_iters[idx] = iter(self.orig_data_iters[idx])

    def __iter__(self):
        while True:
            dataset_idx = np.random.choice(len(self.sampling_probs), p=self.sampling_probs)
            try:
                data = next(self.data_iters[dataset_idx])
            except StopIteration:  # If the iterator is exhausted
                self.reset_iterator(dataset_idx)  # Reset it
                data = next(self.data_iters[dataset_idx])

            mod_dict = make_empty_mod_dict(self.modality_info)
            mod_dict.update(data)
            yield mod_dict


def build_mixture_dataloader(data_iters, weights, modality_info, batch_size, num_workers, epoch_size, num_gpus):
    mixture_pipe = wds.DataPipeline(
        MixtureDataset(data_iters, weights, modality_info),
        wds.batched(batch_size, collation_fn=default_collate, partial=False),
    ).with_epoch(epoch_size // (num_gpus * num_workers * batch_size)) # Pre-define iterator length
    
    mixture_loader = wds.WebLoader(mixture_pipe, num_workers=num_workers, batch_size=None)
    
    return mixture_loader
