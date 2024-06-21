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

import torch
import yaml

import fourm.utils as utils

from fourm.data import (CenterCropImageAugmenter, EmptyAugmenter,
                  PreTokenizedImageAugmenter,RandomCropImageAugmenter, build_fm_pretraining_dataset,
                  build_huggingface_pretraining_dataloader,
                  build_wds_fm_pretraining_dataloader)
from fourm.data.modality_transforms import CaptionTransform
from fourm.data.modality_info import MODALITY_TRANSFORMS


def setup_sampling_mod_info(dataset_config, modality_info):
    # Subset of modality info for each dataset

    # Input and output modalities for one dataset
    in_domains = sorted(dataset_config['in_domains'].split('-'))
    out_domains = sorted(dataset_config['out_domains'].split('-'))
    all_domains = sorted(list(set(in_domains) | set(out_domains)))

    mod_info = copy.deepcopy(modality_info)
    mod_info = {mod: mod_info[mod] for mod in all_domains}

    # Dirichlet concentration parameter (Alpha)
    if dataset_config.get('alphas_config', None) is None:
        for mod in mod_info:
            mod_info[mod]["input_alphas"] = [0.]
            mod_info[mod]["target_alphas"] = [0.]

        if 'input_alphas' in dataset_config:
            input_alphas = dataset_config['input_alphas'].split('-')
            if len(input_alphas) == 1:
                input_alphas = [float(input_alphas[0])] * len(in_domains)
            else:
                input_alphas = [float(alpha) for alpha in input_alphas]
            for mod, alpha in zip(in_domains, input_alphas):
                mod_info[mod]['input_alphas'] = [alpha]

        if 'target_alphas' in dataset_config:
            target_alphas = dataset_config['target_alphas'].split('-')
            if len(target_alphas) == 1:
                target_alphas = [float(target_alphas[0])] * len(out_domains)
            else:
                target_alphas = [float(alpha) for alpha in target_alphas]
            for mod, alpha in zip(out_domains, target_alphas):
                mod_info[mod]["target_alphas"] = [alpha]

        sampling_weights = None
    else:
        print(f"Loading alphas config from: {dataset_config['alphas_config']}")
        with open(dataset_config['alphas_config'], "r") as f:
            alphas_config = yaml.safe_load(f)

        if 'sampling_weights' in alphas_config:
            sampling_weights = alphas_config['sampling_weights']
            alphas_config = alphas_config['alphas_mixture']
        else:
            sampling_weights = None
        
        for mod in mod_info:
            mod_info[mod]["input_alphas"] = alphas_config[mod]["input_alphas"]
            mod_info[mod]["target_alphas"] = alphas_config[mod]["target_alphas"]
            if modality_info[mod]['type'] in ['seq', 'seq_emb', 'seq_token']:
                mod_info[mod]['keep'] = alphas_config[mod]['keep']
    
    return mod_info, sampling_weights

def get_train_dataloader(dataset_config, modality_info, sampling_weights, text_tokenizer, input_size, 
                         num_input_tokens, num_target_tokens, min_input_tokens, min_target_tokens,
                         num_tasks, num_workers, dataset_batch_size=None, epoch_size=None):
    
    in_domains = sorted(list(dataset_config['in_domains'].split('-')))
    out_domains = sorted(list(dataset_config['out_domains'].split('-')))
    all_domains = sorted(list(set(in_domains) | set(out_domains)))

    modality_transforms = MODALITY_TRANSFORMS
    if 'caption' in modality_transforms:
        modality_transforms['caption'] = CaptionTransform(
            aligned_captions=dataset_config.get('aligned_captions', True)
        )
    
    if dataset_config['type'] == 'multimodal':

        is_pretokenized = any([modality_info[mod].get('pretokenized', False) for mod in modality_info])
        if is_pretokenized:
            # Multi-modal training data augmentation (uses pre-tokenized data augmentation)
            image_augmenter = PreTokenizedImageAugmenter(
                target_size=input_size, 
                no_aug=(not dataset_config.get('tok_train_aug', True)), 
                main_domain=dataset_config['main_augment_domain']
            )
        else:
            image_augmenter = RandomCropImageAugmenter(
                 target_size=input_size, 
                 hflip=dataset_config.get('hflip'), 
                 crop_scale=tuple(dataset_config.get('crop_scale')),
                 crop_ratio=tuple(dataset_config.get('crop_ratio')),
            )

        # Input and target token ranges
        num_input_tokens = dataset_config.get('num_input_tokens', num_input_tokens)
        num_target_tokens = dataset_config.get('num_target_tokens', num_target_tokens)
        min_input_tokens = dataset_config.get('min_input_tokens', min_input_tokens)
        min_target_tokens = dataset_config.get('min_target_tokens', min_target_tokens)
        min_input_tokens = num_input_tokens if min_input_tokens is None else min_input_tokens
        min_target_tokens = num_target_tokens if min_target_tokens is None else min_target_tokens


        if dataset_config['use_wds']:
            # Using webdataset
            loader = build_wds_fm_pretraining_dataloader(
                data_path=dataset_config['data_path'], all_domains=all_domains,
                modality_info=modality_info, modality_transforms=modality_transforms,
                image_augmenter=image_augmenter, text_tokenizer=text_tokenizer,
                input_tokens_range=(min_input_tokens, num_input_tokens),
                target_tokens_range=(min_target_tokens, num_target_tokens),
                num_gpus=num_tasks, num_workers=num_workers,
                batch_size=dataset_batch_size, epoch_size=epoch_size,
                modality_name_map=dataset_config.get('modality_name_map', None),
                shuffle_buffer_load=dataset_config.get('wds_shuffle_buffer_tar', 1_000),
                shuffle_buffer_repeat=dataset_config.get('wds_shuffle_buffer_repeat', 1_000),
                n_repeats=dataset_config.get('wds_n_repeats', 1),
                sampling_weights=sampling_weights,
            )
        else:
            dataset_train = build_fm_pretraining_dataset(
                data_path=dataset_config['data_path'], 
                all_domains=all_domains, modality_info=modality_info, modality_transforms=modality_transforms,
                image_augmenter=image_augmenter, text_tokenizer=text_tokenizer,
                input_tokens_range=(min_input_tokens, num_input_tokens), 
                target_tokens_range=(min_target_tokens, num_target_tokens)
            )
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=utils.get_rank(), shuffle=True, drop_last=True,
            )
            # DataLoader has batch size 1 as it then gets collated through the Mixture dataloader
            loader = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=1, num_workers=0,
                pin_memory=False, drop_last=True,
                collate_fn=lambda x: x[0],
            )

    elif dataset_config['type'] == 'huggingface':

        # Input and target token ranges
        num_input_tokens = dataset_config.get('num_input_tokens', num_input_tokens)
        num_target_tokens = dataset_config.get('num_target_tokens', num_target_tokens)
        
        if dataset_config.get('use_wds', False):
            raise NotImplementedError('Webdataset not yet implemented for huggingface datasets.')
        else:
            loader = build_huggingface_pretraining_dataloader(
                data_path=dataset_config['data_path'], all_domains=all_domains,
                modality_info=modality_info, modality_transforms=modality_transforms,
                image_augmenter=EmptyAugmenter(), text_tokenizer=text_tokenizer,
                input_tokens_range=(num_input_tokens, num_input_tokens), 
                target_tokens_range=(num_target_tokens, num_target_tokens),
                num_gpus=num_tasks, num_workers=num_workers,
                batch_size=dataset_batch_size, epoch_size=epoch_size,
                split='train', streaming=True, rename_text_to_caption=True, 
                shuffle_buffer_load=dataset_config.get('shuffle_buffer_load', 1_000),
                shuffle_seed=0,
            )
    else:
        raise NotImplementedError(f'Dataset type {dataset_config["type"]} not implemented.')

    return loader
    

def cfgs_get(key, val_config, dataset_name, train_configs, default=None):
    """ Try to retrieve a key from the validation set config.
    If it does not exist, default to retrieving it from the train set config
    with the same dataset name.
    """
    return val_config.get(key, train_configs[dataset_name].get(key, default))


def get_val_dataloader(dataset_config, dataset_name, train_configs, modality_info, sampling_weights, text_tokenizer, 
                       input_size, num_input_tokens, num_target_tokens, min_input_tokens, min_target_tokens,
                       fixed_eval, fixed_eval_input_tokens, fixed_eval_target_tokens, 
                       dist_eval, num_tasks, num_workers, batch_size, pin_mem):
    
    in_domains = sorted(list(cfgs_get('in_domains', dataset_config, dataset_name, train_configs).split('-')))
    out_domains = sorted(list(cfgs_get('out_domains', dataset_config, dataset_name, train_configs).split('-')))
    all_domains = sorted(list(set(in_domains) | set(out_domains)))

    modality_transforms = MODALITY_TRANSFORMS
    if 'caption' in modality_transforms:
        modality_transforms['caption'] = CaptionTransform(
            aligned_captions=cfgs_get('aligned_captions', dataset_config, dataset_name, train_configs, True)
        )

    dataset_type = cfgs_get('type', dataset_config, dataset_name, train_configs)

    if dataset_type == 'multimodal':

        main_augment_domain = cfgs_get('main_augment_domain', dataset_config, dataset_name, train_configs)
        is_pretokenized = any([modality_info[mod].get('pretokenized', False) for mod in modality_info])
        if is_pretokenized:
            eval_image_augmenter = PreTokenizedImageAugmenter(
                target_size=input_size, no_aug=True, main_domain=main_augment_domain
            )
        else:
            eval_image_augmenter = CenterCropImageAugmenter(
                target_size=input_size, main_domain=main_augment_domain
            )
            

        if fixed_eval:
            input_tokens_range=(fixed_eval_input_tokens, fixed_eval_input_tokens)
            target_tokens_range=(fixed_eval_target_tokens, fixed_eval_target_tokens)
        else:
            # Input and target token ranges
            num_input_tokens = dataset_config.get('num_input_tokens', num_input_tokens)
            num_target_tokens = dataset_config.get('num_target_tokens', num_target_tokens)
            min_input_tokens = dataset_config.get('min_input_tokens', min_input_tokens)
            min_target_tokens = dataset_config.get('min_target_tokens', min_target_tokens)
            min_input_tokens = num_input_tokens if min_input_tokens is None else min_input_tokens
            min_target_tokens = num_target_tokens if min_target_tokens is None else min_target_tokens
            input_tokens_range = (min_input_tokens, num_input_tokens)
            target_tokens_range = (min_target_tokens, num_target_tokens)

        dataset_val = build_fm_pretraining_dataset(
            data_path=cfgs_get('data_path', dataset_config, dataset_name, train_configs), 
            all_domains=all_domains, modality_info=modality_info, modality_transforms=modality_transforms,
            image_augmenter=eval_image_augmenter, text_tokenizer=text_tokenizer,
            input_tokens_range=input_tokens_range, target_tokens_range=target_tokens_range
        )

        print("Warning: Eval stats may vary slightly as the masking applied on images is random.")
        if dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=utils.get_rank(), shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        loader = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=False,
        )

    elif dataset_type == 'huggingface':

        if fixed_eval:
            input_tokens_range=(fixed_eval_input_tokens, fixed_eval_input_tokens)
            target_tokens_range=(fixed_eval_target_tokens, fixed_eval_target_tokens)
        else:
            # Input and target token ranges
            num_input_tokens = dataset_config.get('num_input_tokens', num_input_tokens)
            num_target_tokens = dataset_config.get('num_target_tokens', num_target_tokens)
            input_tokens_range = (num_input_tokens, num_input_tokens)
            target_tokens_range = (num_target_tokens, num_target_tokens)

        loader = build_huggingface_pretraining_dataloader(
            data_path=cfgs_get('data_path', dataset_config, dataset_name, train_configs), 
            all_domains=all_domains, modality_info=modality_info, modality_transforms=modality_transforms,
            image_augmenter=EmptyAugmenter(), text_tokenizer=text_tokenizer,
            input_tokens_range=input_tokens_range, target_tokens_range=target_tokens_range,
            num_gpus=num_tasks, num_workers=num_workers,
            batch_size=batch_size, epoch_size=None,
            split='validation', streaming=True, rename_text_to_caption=True, 
            shuffle_buffer_load=cfgs_get('shuffle_buffer_load', dataset_config, dataset_name, train_configs, 1_000),
            shuffle_seed=0,
        )

    else:
        raise NotImplementedError(f'Dataset type {dataset_type} not implemented.')

    return loader