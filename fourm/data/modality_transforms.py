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
import gzip
import json
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from abc import ABC, abstractmethod

from PIL import Image
import cv2

import albumentations as A
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from einops import rearrange, repeat, reduce

from fourm.utils import to_2tuple
from fourm.utils.data_constants import (IMAGENET_DEFAULT_MEAN,
                                  IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN,
                                  IMAGENET_SURFACE_NORMAL_STD, IMAGENET_SURFACE_NORMAL_MEAN,
                                  IMAGENET_INCEPTION_STD, SEG_IGNORE_INDEX, PAD_MASK_VALUE)


# The @-symbol is used to specify the resolution of a modality. Syntax: modality@resolution
def get_transform_key(mod_name):
    return mod_name.split('@')[0]

def get_transform_resolution(mod_name, default_resolution, to_tuple=True):
    res = int(mod_name.split('@')[1]) if '@' in mod_name else default_resolution
    return to_2tuple(res) if to_tuple else res

def get_transform(mod_name, transforms_dict):
    return transforms_dict.get(get_transform_key(mod_name), IdentityTransform())

def get_pil_resample_mode(resample_mode: str):
    """
    Returns the PIL resampling mode for the given resample mode string.

    Args:
        resample_mode: Resampling mode string
    """
    if resample_mode is None:
        return None
    elif resample_mode == "bilinear":
        return Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR
    elif resample_mode == "bicubic":
        return Image.Resampling.BICUBIC if hasattr(Image, 'Resampling') else Image.BICUBIC
    elif resample_mode == "nearest":
        return Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST
    else:
        raise ValueError(f"Resample mode {resample_mode} is not supported.")

class UnifiedDataTransform(object):
    def __init__(self, transforms_dict, image_augmenter, resample_mode: str = None, add_sizes: bool = False, **kwargs):
        """Unified data augmentation for FourM

        Args:
            transforms_dict (dict): Dict of transforms for each modality
            image_augmenter (AbstractImageAugmenter): Image augmenter
            resample_mode (str, optional): Resampling mode for PIL images (default: None -> uses default resampling mode for data type)
                One out of ["bilinear", "bicubic", "nearest", None].
            add_sizes (bool, optional): Whether to add crop coordinates and original size to the output dict
        """

        self.transforms_dict = transforms_dict
        self.image_augmenter = image_augmenter
        self.resample_mode = resample_mode
        self.add_sizes = add_sizes

    def unified_image_augment(self, mod_dict, crop_settings):
        """Apply the image augmenter to all modalities where it is applicable

        Args:
            mod_dict (dict): Dict of modalities
            crop_settings (dict): Crop settings

        Returns:
            dict: Transformed dict of modalities
        """

        crop_coords, flip, orig_size, target_size, rand_aug_idx = self.image_augmenter(mod_dict, crop_settings)
        
        mod_dict = {
            k: self.transforms_dict[get_transform_key(k)].image_augment(
                v, crop_coords=crop_coords, flip=flip, orig_size=orig_size, 
                target_size=get_transform_resolution(k, target_size), rand_aug_idx=rand_aug_idx,
                resample_mode=self.resample_mode
            )
            for k, v in mod_dict.items()
        }

        if self.add_sizes:
            mod_dict["crop_coords"] = torch.tensor(crop_coords)
            mod_dict["orig_size"] = torch.tensor(orig_size)

        return mod_dict

    def __call__(self, mod_dict):
        """Apply the augmentation to a dict of modalities (both image based and sequence based modalities)

        Args:
            mod_dict (dict): Dict of modalities

        Returns:
            dict: Transformed dict of modalities
        """
        crop_settings = mod_dict.pop("crop_settings", None)

        mod_dict = {k: get_transform(k, self.transforms_dict).preprocess(v) for k, v in mod_dict.items()}

        mod_dict = self.unified_image_augment(mod_dict, crop_settings)

        mod_dict = {k: get_transform(k, self.transforms_dict).postprocess(v) for k, v in mod_dict.items()}

        return mod_dict

    def __repr__(self):
        repr = "(UnifiedDataAugmentation,\n"
        repr += ")"
        return repr


class AbstractTransform(ABC):

    @abstractmethod
    def load(self, sample):
        pass

    @abstractmethod
    def preprocess(self, sample):
        pass

    @abstractmethod
    def image_augment(self, v, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        pass

    @abstractmethod
    def postprocess(self, v):
        pass


class ImageTransform(AbstractTransform):

    @staticmethod
    def pil_loader(path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        # with open(path, 'rb') as f:
        #     img = Image.open(f)
        img = Image.open(path)
        return img


    @staticmethod
    def image_hflip(img: Image, flip: bool):
        """Crop and resize an image

        :param img: Image to crop and resize
        :param flip: Whether to flip the image
        :return: Flipped image (if flip = True)
        """
        if flip:
            img = TF.hflip(img)
        return img

    @staticmethod
    def image_crop_and_resize(img: Image, crop_coords: Tuple, target_size: Tuple, resample_mode: str = None):
        """Crop and resize an image

        :param img: Image to crop and resize
        :param crop_coords: Coordinates of the crop (top, left, h, w)
        :param target_size: Coordinates of the resize (height, width)
        :return: Cropped and resized image
        """

        top, left, h, w = crop_coords
        resize_height, resize_width = target_size
        img = TF.crop(img, top, left, h, w)
        resample_mode = get_pil_resample_mode(resample_mode)
        img = img.resize((resize_height, resize_width), resample=resample_mode)
        return img


class RGBTransform(ImageTransform):

    def __init__(self, imagenet_default_mean_and_std=True, color_jitter=False, color_jitter_strength=0.5):
        self.rgb_mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        self.rgb_std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        self.color_jitter = color_jitter
        self.color_jitter_transform = self.random_color_jitter(color_jitter_strength)

    def random_color_jitter(self, strength=0.5):
        # Color Jitter from Pix2Seq and SimCLR
        # Source: https://github.com/google-research/pix2seq/blob/main/data/data_utils.py#L114
        t = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.8 * strength, contrast=0.8 * strength, saturation=0.8 * strength, hue=0.2 * strength)], p=0.8),
            T.RandomApply([T.Grayscale(num_output_channels=3)], p=0.2),
        ])

        return t

    def rgb_to_tensor(self, img):
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.rgb_mean, std=self.rgb_std)
        return img

    def load(self, path):
        # TODO: Instead of converting to RGB here, do it either in the preprocess or the postprocess step. Makes it compatible with wds dataloading.
        sample = self.pil_loader(path)
        return sample

    def preprocess(self, sample):
        sample = sample.convert('RGB')
        
        if self.color_jitter:
            sample = self.color_jitter_transform(sample)

        return sample

    def image_augment(self, img, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple, 
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        img = self.image_crop_and_resize(img, crop_coords, target_size, resample_mode=resample_mode)
        img = self.image_hflip(img, flip)
        return img

    def postprocess(self, sample):
        sample = self.rgb_to_tensor(sample)
        return sample


class DepthTransform(ImageTransform):

    def __init__(self, standardize_depth=True):
        self.standardize_depth = standardize_depth

    def depth_to_tensor(self, img):
        img = torch.Tensor( img / (2 ** 16 - 1.0) )
        img = img.unsqueeze(0)  # 1 x H x W
        if self.standardize_depth:
            img = self.truncated_depth_standardization(img)
        return img

    @staticmethod
    def truncated_depth_standardization(depth, thresh: float = 0.1):
        """Truncated depth standardization

        :param depth: Depth map
        :param thresh: Threshold
        :return: Robustly standardized depth map
        """
        # Flatten depth and remove bottom and top 10% of values
        trunc_depth = torch.sort(depth.reshape(-1), dim=0)[0]
        trunc_depth = trunc_depth[int(thresh * trunc_depth.shape[0]): int((1 - thresh) * trunc_depth.shape[0])]
        return (depth - trunc_depth.mean()) / torch.sqrt(trunc_depth.var() + 1e-6)

    def load(self, path):
        sample = self.pil_loader(path)
        return sample

    def preprocess(self, sample):
        return sample

    def image_augment(self, img, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple, 
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        img = self.image_crop_and_resize(img, crop_coords, target_size, resample_mode=resample_mode)
        img = self.image_hflip(img, flip)
        return img

    def postprocess(self, sample):
        sample = np.array(sample)
        sample = self.depth_to_tensor(sample)
        return sample


class NormalTransform(ImageTransform):

    def __init__(self, standardize_surface_normals=False):
        self.normal_mean = (0.5, 0.5, 0.5) if not standardize_surface_normals else IMAGENET_SURFACE_NORMAL_MEAN
        self.normal_std = (0.5, 0.5, 0.5) if not standardize_surface_normals else IMAGENET_SURFACE_NORMAL_STD

    def normal_to_tensor(self, img):
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.normal_mean, std=self.normal_std)
        return img

    def load(self, path):
        sample = self.pil_loader(path)
        return sample

    def preprocess(self, sample):
        return sample

    def image_hflip(self, img: Image, flip: bool):
        if flip:
            img = TF.hflip(img)
            flipped_np = np.array(img)
            flipped_np[:, :, 0] = 255 - flipped_np[:, :, 0]
            img = Image.fromarray(flipped_np)

        return img

    def image_augment(self, img, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        img = self.image_crop_and_resize(img, crop_coords, target_size, resample_mode=resample_mode)
        img = self.image_hflip(img, flip)
        return img

    def postprocess(self, sample):
        sample = self.normal_to_tensor(sample)
        return sample
    

class SemsegTransform(ImageTransform):

    def __init__(self, scale_factor=1.0, shift_idx_by_one=False, id_mapping: Optional[Dict] = None, select_channel=None):
        self.scale_factor = scale_factor
        self.shift_idx_by_one = shift_idx_by_one
        self.id_mapping = id_mapping
        self.select_channel = select_channel

    def map_semseg_values(self, sample):
        sample = np.asarray(sample)
        mapping_fn = lambda x: self.id_mapping.get(x, x)
        sample = np.vectorize(mapping_fn)(sample)
        sample = Image.fromarray(sample, mode='P')
        return sample

    def semseg_to_tensor(self, img):
        # Rescale to scale factor
        if self.scale_factor != 1.0:
            target_height, target_width = int(img.height * self.scale_factor), int(img.width * self.scale_factor)
            img = img.resize((target_width, target_height))
        # Using pil_to_tensor keeps it in uint8, to_tensor converts it to float (rescaled to [0, 1])
        img = TF.pil_to_tensor(img).to(torch.long).squeeze(0)
        # 255->0, 254->0, all else shifted up by one
        return img

    def load(self, path):
        sample = self.pil_loader(path)
        if self.select_channel is not None:
            sample = sample.split()[self.select_channel]
        return sample

    def preprocess(self, sample):
        sample = sample.convert('P')

        if self.id_mapping is not None:
            sample = self.map_semseg_values(sample)

        if self.shift_idx_by_one:
            sample = np.asarray(sample)
            sample = sample + 1
            sample = Image.fromarray(sample, mode='P')

        return sample

    def image_augment(self, img, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        # Value for padding with TF.crop is always 0.
        # Override resampling mode to 'nearest' for semseg
        img = self.image_crop_and_resize(img, crop_coords, target_size, resample_mode='nearest')
        img = self.image_hflip(img, flip)
        return img

    def postprocess(self, sample):
        img = self.semseg_to_tensor(sample)
        return img


class MaskTransform(ImageTransform):

    def __init__(self, mask_pool_size=1):
        assert isinstance(mask_pool_size, int)
        self.mask_pool_size = mask_pool_size # Use to expand masks

    def mask_to_tensor(self, img):
        mask = TF.to_tensor(img)
        if self.mask_pool_size > 1:
            mask = reduce(mask, 'c (h1 h2) (w1 w2) -> c h1 w1', 'min', h2=self.mask_pool_size, w2=self.mask_pool_size)
            mask = repeat(mask, 'c h1 w1 -> c (h1 h2) (w1 w2)', h2=self.mask_pool_size, w2=self.mask_pool_size)
        return (mask == 1.0)

    def load(self, path):
        sample = self.pil_loader(path)
        return sample

    def preprocess(self, sample):
        return sample

    def image_augment(self, img, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        # Override resampling mode to 'nearest' for masks
        img = self.image_crop_and_resize(img, crop_coords, target_size, resample_mode='nearest')
        img = self.image_hflip(img, flip)
        return img

    def postprocess(self, sample):
        sample = self.mask_to_tensor(sample)
        return sample


class TokTransform(AbstractTransform):

    def __init__(self):
        pass

    def load(self, path):
        sample = np.load(path).astype(int)
        return sample

    def preprocess(self, sample):
        return sample

    def image_augment(self, v, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        if rand_aug_idx is None:
            raise ValueError("Crop settings / augmentation index are missing but a pre-tokenized modality is being used")
        v = torch.tensor(v[rand_aug_idx])
        return v

    def postprocess(self, sample):
        return sample


class DetectionTransform(AbstractTransform):

    def __init__(self, det_threshold=0.6, det_max_instances=None, bbox_order='dist_to_orig', coord_bins=1000, min_visibility=0.0, return_raw=False):
        self.det_threshold = det_threshold
        self.det_max_instances = det_max_instances
        self.coord_bins = coord_bins
        self.min_visibility = min_visibility
        self.return_raw = return_raw

        if bbox_order == 'area':
            self.bbox_order = self.order_bboxes_by_area
        elif bbox_order == 'score':
            self.bbox_order = self.order_bboxes_by_score
        elif bbox_order == 'random':
            self.bbox_order = self.shuffle_bboxes
        else:
            self.bbox_order = self.order_bboxes_by_dist_to_orig

    @staticmethod
    def order_bboxes_by_area(bboxes):
        return sorted(bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)

    @staticmethod
    def order_bboxes_by_dist_to_orig(bboxes):
        return sorted(bboxes, key=lambda x: x[0] ** 2 + x[1] ** 2)

    @staticmethod
    def order_bboxes_by_score(bboxes):
        return sorted(bboxes, key=lambda x: x[5], reverse=True)

    @staticmethod
    def shuffle_bboxes(bboxes):
        return sorted(bboxes, key=lambda x: random.random())

    def convert_detection_instance(self, instances):
        """Convert instances dict to list of lists where each list takes the form:
        [xmin, ymin, xmax, ymax, class_name, score]
        """

        instances = [inst['boxes'] + [inst['class_name'], inst['score']] for inst in instances if inst['score'] >= self.det_threshold]
        return instances

    def bboxes_hflip(self, bboxes: List[Tuple], image_size: Tuple, flip: bool):
        image_height, image_width = image_size
        if flip:
            bboxes = [tuple(A.bbox_hflip(bbox[:4], rows=image_height, cols=image_width)) + tuple(bbox[4:])
                      for bbox in bboxes]

        return bboxes

    def bboxes_crop_and_resize(self, bboxes: List[Tuple], crop_coords: Tuple, orig_size: Tuple):
        """Crop and resize bounding boxes

        Args:
            bboxes: Bounding boxes to crop and resize
            crop_coords: Coordinates of the crop (top, left, h, w)
            orig_size: Size of the original image

        Returns:
            Cropped and resized bounding boxes
        """
        orig_height, orig_width = orig_size
        top, left, h, w = crop_coords
        xmin, ymin, xmax, ymax = left, top, left + w, top + h
        bboxes = [tuple(A.bbox_crop(bbox[:4], x_min=xmin, y_min=ymin, x_max=xmax, y_max=ymax, rows=orig_height,
                                    cols=orig_width)) + tuple(bbox[4:])
                  for bbox in bboxes]
        bboxes = A.core.bbox_utils.filter_bboxes(bboxes, rows=h, cols=w, min_visibility=self.min_visibility)
        # No need to resize, bounding boxes in albumentations format are scale invariant

        return bboxes

    def order_and_filter_bboxes(self, bboxes):
        if self.det_max_instances is not None and len(bboxes) > self.det_max_instances:
            bboxes = self.order_bboxes_by_score(bboxes)[:self.det_max_instances]

        return self.bbox_order(bboxes)

    def convert_bboxes_to_string(self, bboxes: List[Tuple]):
        """Convert bounding boxes to a string

        Args:
            bboxes: Bounding boxes

        Returns:
            String representation of the bounding boxes
        """
        # Remove score, quantize coordinates
        bins = self.coord_bins

        bboxes = [
            [
                f"xmin={round(xmin * (bins - 1))}",
                f"ymin={round(ymin * (bins - 1))}",
                f"xmax={round(xmax * (bins - 1))}",
                f"ymax={round(ymax * (bins - 1))}",
                cls,
            ]
            for (xmin, ymin, xmax, ymax, cls, score) in bboxes
        ]
        # Convert each bounding box to a string
        bboxes = [' '.join(b) for b in bboxes]
        # Convert the list to a str
        return ' '.join(bboxes)

    def load(self, path):
        with open(path, 'r') as f:
            sample = json.load(f)

        return sample

    def preprocess(self, sample):
        instances = sample['instances']
        return self.convert_detection_instance(instances)

    def image_augment(self, bboxes: List[Tuple], crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple, 
                      rand_aug_idx=None, resample_mode: str = None):
        bboxes = self.bboxes_crop_and_resize(bboxes, crop_coords, orig_size)
        bboxes = self.bboxes_hflip(bboxes, target_size, flip)
        bboxes = self.order_and_filter_bboxes(bboxes)
        return bboxes

    def postprocess(self, bboxes):
        if self.return_raw:
            return bboxes
        bboxes = self.convert_bboxes_to_string(bboxes)
        return bboxes


class CaptionTransform(AbstractTransform):

    def __init__(self, aligned_captions=True, no_aug=False):
        self.aligned_captions = aligned_captions
        self.no_aug = no_aug

    def load(self, path):
        # Caption can either be stored as .txt or .json.gz (in which case it's a list of dicts)
        if path.endswith('.txt'):
            sample = Path(path).read_text()
        elif path.endswith('.json'):
            with open(path, 'r') as f:
                sample = json.load(f)
        elif path.endswith('.json.gz'):
            with gzip.open(path, 'rb') as f:
                sample = json.load(f)
        return sample

    def preprocess(self, sample):
        return sample

    def image_augment(self, val, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple, 
                      rand_aug_idx: Optional[int], resample_mode: str = None):

        if isinstance(val, list) or isinstance(val, tuple):
            if self.aligned_captions:
                val = val[0] if rand_aug_idx is None else val[rand_aug_idx]
            else:
                val = random.choice(val) if not self.no_aug else val[0]

        if isinstance(val, dict):
            # If each caption is saved as a dict, extract the string
            val = val["caption"]
        assert isinstance(val, str)

        return val

    def postprocess(self, sample):
        return sample


class CropSettingsTransform(AbstractTransform):

    def load(self, path):
        sample = np.load(path)
        return sample

    def preprocess(self, sample):
        raise NotImplementedError("CropSettingsTransform does not support preprocessing")

    def image_augment(self, val, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple, 
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        raise NotImplementedError("CropSettingsTransform is not meant to be used for image augmentation")

    def postprocess(self, sample):
        raise NotImplementedError("CropSettingsTransform does not support postprocessing")


class IdentityTransform(AbstractTransform):

    def load(self, path):
        raise NotImplementedError("IdentityTransform does not support loading")

    def preprocess(self, sample):
        return sample

    def image_augment(self, val, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple, 
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        return val

    def postprocess(self, sample):
        return sample


class JSONTransform(AbstractTransform):

    def load(self, path):
        if path.endswith('.json'):
            with open(path, 'r') as f:
                sample = json.load(f)
        elif path.endswith('.json.gz'):
            with gzip.open(path, 'rb') as f:
                sample = json.load(f)
        return sample

    def preprocess(self, sample):
        return sample

    def image_augment(self, val, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple, 
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        return val

    def postprocess(self, sample):
        return sample