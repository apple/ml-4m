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
import random
from abc import ABC, abstractmethod

import numpy as np
import torchvision

from fourm.utils import to_2tuple


class AbstractImageAugmenter(ABC):
    """Abstract class for image augmenters.
    """

    @abstractmethod
    def __call__(self, mod_dict, crop_settings):
        pass


class RandomCropImageAugmenter(AbstractImageAugmenter):

    def __init__(self, target_size=224, hflip=0.5, crop_scale=(0.2, 1.0), crop_ratio=(0.75, 1.3333), main_domain='rgb'):

        self.target_size = to_2tuple(target_size)
        self.hflip = hflip
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.main_domain = main_domain

    def __call__(self, mod_dict, crop_settings):

        if crop_settings is not None:
            raise ValueError("Crop settings are provided but not used by this augmenter.")

        image = mod_dict[self.main_domain] if self.main_domain is not None else mod_dict[list(mod_dict.keys())[0]]
        # With torchvision 0.13+, can also be: orig_size = TF.get_dimensions(image)
        orig_width, orig_height = image.size
        orig_size = (orig_height, orig_width)

        top, left, h, w = torchvision.transforms.RandomResizedCrop.get_params(
            image, scale=self.crop_scale, ratio=self.crop_ratio
        )
        crop_coords = top, left, h, w
        flip = random.random() < self.hflip
        rand_aug_idx = None

        return crop_coords, flip, orig_size, self.target_size, rand_aug_idx

class NoImageAugmenter(AbstractImageAugmenter): # this is for non-image modalities like poses where we don't do any augs, e.g. during tokenization 

    def __init__(self, no_aug=True, main_domain='human_poses'):
        self.target_size = None #to_2tuple(target_size)
        self.no_aug = no_aug
        self.main_domain = main_domain

    def __call__(self, mod_dict, crop_settings):
        # # With torchvision 0.13+, can also be: orig_size = TF.get_dimensions(image)
        orig_size = (224, 224)

        rand_aug_idx = 0 
        top, left, h, w, flip = 0, 0, 224, 224, 0 
        crop_coords = (top, left, h, w)

        return crop_coords, flip, orig_size, self.target_size, rand_aug_idx

class PreTokenizedImageAugmenter(AbstractImageAugmenter):

    def __init__(self, target_size, no_aug=False, main_domain='rgb'):
        self.target_size = to_2tuple(target_size)
        self.no_aug = no_aug
        self.main_domain = main_domain

    def __call__(self, mod_dict, crop_settings):
        # With torchvision 0.13+, can also be: orig_size = TF.get_dimensions(image)
        if self.main_domain in mod_dict and 'tok' not in self.main_domain:
            image = mod_dict[self.main_domain] if self.main_domain is not None else mod_dict[list(mod_dict.keys())[0]]
            orig_width, orig_height = image.size
            orig_size = (orig_height, orig_width)
        else:
            orig_size = None

        rand_aug_idx = 0 if self.no_aug else np.random.randint(len(crop_settings))
        top, left, h, w, flip = crop_settings[rand_aug_idx]
        crop_coords = (top, left, h, w)

        return crop_coords, flip, orig_size, self.target_size, rand_aug_idx


class CenterCropImageAugmenter(AbstractImageAugmenter):
    def __init__(self, target_size, hflip=0.0, main_domain='rgb'):
        self.target_size = to_2tuple(target_size)
        self.hflip = hflip
        self.main_domain = main_domain

    def __call__(self, mod_dict, crop_settings=None):
        image = mod_dict[self.main_domain] if self.main_domain is not None else mod_dict[list(mod_dict.keys())[0]]
        orig_width, orig_height = image.size
        orig_size = (orig_height, orig_width)

        if orig_height > orig_width:
            h = w = orig_width
            top = (orig_height - orig_width) // 2
            left = 0
        else:
            h = w = orig_height
            top = 0
            left = (orig_width - orig_height) // 2

        crop_coords = (top, left, h, w)
        flip = random.random() < self.hflip
        rand_aug_idx = None

        return crop_coords, flip, orig_size, self.target_size, rand_aug_idx


class PaddingImageAugmenter(AbstractImageAugmenter):
    def __init__(self, target_size, hflip=0.0, main_domain='rgb'):
        self.target_size = to_2tuple(target_size)
        self.hflip = hflip
        self.main_domain = main_domain

    def __call__(self, mod_dict, crop_settings):
        image = mod_dict[self.main_domain] if self.main_domain is not None else mod_dict[list(mod_dict.keys())[0]]
        orig_width, orig_height = image.size
        orig_size = (orig_height, orig_width)

        h = w = max(orig_width, orig_height)
        top = left = 0
        crop_coords = (top, left, h, w)
        flip = random.random() < self.hflip
        rand_aug_idx = None

        return crop_coords, flip, orig_size, self.target_size, rand_aug_idx


class ScaleJitteringImageAugmenter(AbstractImageAugmenter):
    def __init__(self, target_size, hflip=0.0, scale=(0.1, 2.0), main_domain='rgb'):
        self.target_size = to_2tuple(target_size)
        self.hflip = hflip
        self.scale = scale
        self.main_domain = main_domain

    def scale_jitter(self, orig_height, orig_width):
        rand_scale = np.random.uniform(self.scale[0], self.scale[1])
        max_hw = max(orig_height, orig_width)
        h = w = round(max_hw / rand_scale)
        top = round(max(0, np.random.uniform(0, orig_height - h)))
        left = round(max(0, np.random.uniform(0, orig_width - w)))

        return top, left, h, w

    def __call__(self, mod_dict, crop_settings):

        if crop_settings is not None:
            raise ValueError("Crop settings are provided but not used by this augmenter.")

        image = mod_dict[self.main_domain] if self.main_domain is not None else mod_dict[list(mod_dict.keys())[0]]
        # With torchvision 0.13+, can also be: orig_size = TF.get_dimensions(image)
        orig_width, orig_height = image.size
        orig_size = (orig_height, orig_width)

        crop_coords = self.scale_jitter(orig_height, orig_width)
        flip = random.random() < self.hflip
        rand_aug_idx = None

        return crop_coords, flip, orig_size, self.target_size, rand_aug_idx


class EmptyAugmenter(AbstractImageAugmenter):
    def __init__(self):
        pass

    def __call__(self, mod_dict, crop_settings):
        return None, None, None, None, None