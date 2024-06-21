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


class SAMInstanceTransform(AbstractTransform):

    def __init__(self, mask_size=64, max_instance_n=20, bbox_area_threshold=0.0005):
        self.mask_size = mask_size
        self.max_instance_n = max_instance_n
        self.bbox_area_threshold = bbox_area_threshold

    def get_bbox(self, instance):
        """ Gets bounding box of the given instance
        """
        min_h, max_h =  instance[:,:,1].min(), instance[:,:,1].max()
        min_w, max_w =  instance[:,:,0].min(), instance[:,:,0].max()
        return [min_h, min_w, max_h, max_w]

    def extend_instance_points(self, instance, border_fn):
        """ Given an instance and a border function `border_fn`, extends the instance points with crossing points between the instance and 
        the crop borders. The crossing points are obtained using border_fn.
        """
        p = instance[:,0]
        p_next = np.roll(p, (-1), axis=(0))
        final_points = []
        for x, xn in zip(p, p_next):
            final_points.append(x)
            for r in border_fn(x, xn):
                final_points.append(r.astype(np.int32))
        p = np.stack(final_points)
        return p[:,None]

    def remove_redundant_lines(self, orig_instance, instance):
        """ Removes the redundant lines added during cropping.
        """
        final_points = []
        for p in instance:
            distance = cv2.pointPolygonTest(orig_instance, (p[0,0].item(), p[0,1].item()), measureDist=True)
            if distance >= 0:
                final_points.append(p[0])
        return np.stack(final_points)[:,None]

    def get_border_functions(self, crop_points):
        """ Creates and returns a function `fn` using crop region coordinates given in crop_points.
        `fn` receives two input points x and xn and returns all the crossing points between the line connecting
        x and xn, and the borders of the cropping rectangle.
        """
        p = crop_points[:,0]
        p_next = np.roll(p, (-1), axis=(0))
        def fn(x, xn):
            output = []
            c_diff = p_next - p
            x_diff = x - xn
            for diff, c in zip(c_diff, p):
                A = np.array([
                        [diff[0], x_diff[0]],
                        [diff[1], x_diff[1]]
                    ])
                b = x - c
                try:
                    lmbda = np.linalg.solve(A, b)
                    if 0 <= lmbda[0] <= 1 and 0 <= lmbda[1] <= 1:
                        output.append(lmbda[1] * xn + (1-lmbda[1]) * x)
                except:
                    continue
            return output
        return fn

    def crop_sample(self, sample, crop_coords):
        """ Crop the sample using crop coordinates.
        """
        top, left, h, w = crop_coords
        crop_region = (left, top, left + w, top + h)
        crop_points = np.array([
            [crop_region[0], crop_region[1]],
            [crop_region[2], crop_region[1]],
            [crop_region[2], crop_region[3]],
            [crop_region[0], crop_region[3]],
        ])[:,None]
        border_functions = self.get_border_functions(crop_points)
        cropped_sample = []
        for instance in sample:
            instance = self.extend_instance_points(instance, border_functions)
            filter_condition = (
                (instance[:, :, 0] > crop_region[0]) &
                (instance[:, :, 0] < crop_region[2]) &
                (instance[:, :, 1] > crop_region[1]) &
                (instance[:, :, 1] < crop_region[3])
            )
            if not np.any(filter_condition):
                continue
            
            instance_copy = instance.copy()
            instance_copy[:, :, 0] = np.clip(instance[:, :, 0], a_min=crop_region[0], a_max=crop_region[2])
            instance_copy[:, :, 1] = np.clip(instance[:, :, 1], a_min=crop_region[1], a_max=crop_region[3])
            instance_copy = self.remove_redundant_lines(instance, instance_copy)
            instance_copy[:, :, 0] -= crop_region[0]
            instance_copy[:, :, 1] -= crop_region[1]

            cropped_sample.append(instance_copy)
        return cropped_sample
    
    def resize_sample(self, sample, original_size, target_size):
        """ Resize the sample
        """
        width_scale = target_size[1] / original_size[1]
        height_scale = target_size[0] / original_size[0]
        resized_sample = []
        for instance in sample:
            instance_copy = instance.copy()
            instance_copy[:, :, 0] = np.round(width_scale * instance_copy[:, :, 0])
            instance_copy[:, :, 1] = np.round(height_scale * instance_copy[:, :, 1])
            resized_sample.append(instance_copy)
        return resized_sample
    
    def remove_tiny_instances(self, sample, image_size):
        """ Remove instances that have an area ratio smaller than `bbox_area_threshold`.
        """
        filtered_sample = []
        for instance in sample:
            min_h, min_w, max_h, max_w = self.get_bbox(instance)
            bbox_area_ratio = (max_h - min_h) * (max_w - min_w) / (image_size[0] * image_size[1])
            if bbox_area_ratio < self.bbox_area_threshold:
                continue
            filtered_sample.append(instance)
        return filtered_sample

    def hflip(self, sample, width):
        """ Horizontal flipping the instances in a sample.
        """
        flipped_sample = []
        for instance in sample:
            instance_copy = instance.copy()
            instance_copy[:, :, 0] = width - instance_copy[:, :, 0]
            flipped_sample.append(instance_copy)
        return flipped_sample
        
    def get_binary_masks(self, sample):
        """ Creates the binary mask of each instance in the sample.
        """
        if self.max_instance_n is None:
            max_instance_n = len(sample)
        else:
            max_instance_n = self.max_instance_n
        masks = np.zeros((max_instance_n, self.mask_size, self.mask_size)) 
        bboxes = np.zeros((max_instance_n, 4))
        valid = np.full(max_instance_n, False)
        for i, instance in enumerate(sample):
            bbox = self.get_bbox(instance)
            min_h, min_w, max_h, max_w = bbox
            instance_copy = instance.copy()
            mask = np.zeros((self.mask_size, self.mask_size), dtype=np.uint8)
            instance_copy[:,:,0] = (instance_copy[:,:,0] - min_w) / (max_w - min_w) * self.mask_size
            instance_copy[:,:,1] = (instance_copy[:,:,1] - min_h) / (max_h - min_h) * self.mask_size
            cv2.drawContours(mask, [instance_copy], 0, (255), thickness=cv2.FILLED)
            masks[i] = mask / 255.0
            bboxes[i] = np.array(bbox)
            valid[i] = True
        return masks, bboxes, valid

    def load(self, path):
        sample = np.load(path, allow_pickle=True)
        return sample

    def preprocess(self, sample):
        if self.max_instance_n is None or len(sample) <= self.max_instance_n:
            indecies = np.arange(len(sample))
        else:
            indecies = np.random.choice(len(sample), size=self.max_instance_n, replace=False)
        return [p['points'] for i, p in enumerate(sample) if i in indecies]

    def image_augment(self, v, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        v = self.crop_sample(v, crop_coords)
        _, _, h, w = crop_coords
        v = self.resize_sample(v, (h, w), target_size)
        v = self.remove_tiny_instances(v, target_size)
        if flip:
            v = self.hflip(v, target_size[0])
        return v

    def postprocess(self, sample):
        sample, bboxes, valid = self.get_binary_masks(sample)
        return {
            'instance': torch.from_numpy(sample).to(torch.float32), 
            'bbox': torch.from_numpy(bboxes).to(torch.float32), 
            'valid': torch.from_numpy(valid)
        }


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
        """Convert bounding boxes to a string. 
        xmin, ymin, xmax, ymax are mapped to v0, v1, v2, v3 special tokens.

        Args:
            bboxes: Bounding boxes

        Returns:
            String representation of the bounding boxes
        """
        # Remove score, quantize coordinates
        bins = self.coord_bins

        bboxes = [
            [
                f"v0={round(xmin * (bins - 1))}",
                f"v1={round(ymin * (bins - 1))}",
                f"v2={round(xmax * (bins - 1))}",
                f"v3={round(ymax * (bins - 1))}",
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


class CaptionEmbTransform(AbstractTransform):

    def __init__(self, aligned_captions=True, no_aug=False):
        self.aligned_captions = aligned_captions
        self.no_aug = no_aug

    def load(self, path):
        if path.endswith('.npz'):
            sample = np.load(path)
            sample = {'emb': sample['emb'], 'mask_valid': sample['mask_valid']}
        else:
            raise ValueError(f"Invalid file format for caption embedding: {path}")
        return sample

    def preprocess(self, sample):
        return sample

    def image_augment(self, val, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple, 
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        
        emb = val['emb']
        mask_valid = val['mask_valid'].astype(bool)
        num_sequences = emb.shape[0]

        if num_sequences > 1:
            if self.aligned_captions:
                if rand_aug_idx is None:
                    emb, mask_valid = emb[0], mask_valid[0]
                else:
                    emb, mask_valid = emb[rand_aug_idx], mask_valid[rand_aug_idx]
            else:
                if self.no_aug:
                    emb, mask_valid = emb[0], mask_valid[0]
                else:
                    rand_idx = random.randint(0, num_sequences - 1)
                    emb, mask_valid = emb[rand_idx], mask_valid[rand_idx]
        else:
            emb, mask_valid = emb[0], mask_valid[0]

        emb = emb[mask_valid] # Keep only valid embeddings

        return emb

    def postprocess(self, sample):
        return torch.tensor(sample)
      

class MetadataTransform(AbstractTransform):

    def __init__(self, 
                 special_vmin: int = 0, 
                 special_vmax: int = 999, 
                 shuffle: bool = True, 
                 random_trunc: bool = False, 
                 return_chunks: bool = True, 
                 return_raw: bool = False,
                 image_dim_bin_size: int = 32,):
        """Metadata transform that takes in a metadata dictionary and converts 
        it into a string, or list of strings (for chunked span masking).
        Uses special tokens v1 to denote metadata types, and v0 for their values.

        Args:
            special_vmin: Minimum value for special tokens
            special_vmax: Maximum value for special tokens
            shuffle: Whether to shuffle the metadata order
            random_trunc: Whether to randomly truncate the returned metadata
            return_chunks: Whether to return a list of strings (for chunked span masking),
                or a single string with all metadata concatenated
            return_raw: Whether to return the raw metadata dictionary
        """
        self.special_vmin = special_vmin
        self.special_vmax = special_vmax
        self.shuffle = shuffle
        self.random_trunc = random_trunc
        self.return_chunks = return_chunks
        self.return_raw = return_raw
        self.image_dim_bin_size = image_dim_bin_size

        # Explicit map to make sure that additional entries do not change existing IDs
        # TODO: Make this work with other text tokenizers
        self.metadata_id_map = {
            'original_width': 'v1=0',
            'original_height': 'v1=1',
            'caption_n_chars': 'v1=2',
            'caption_n_words': 'v1=3',
            'caption_n_sentences': 'v1=4',
            'n_humans': 'v1=5',
            'n_sam_instances': 'v1=6',
            'n_coco_instances': 'v1=7',
            'coco_instance_diversity': 'v1=8',
            'colorfulness': 'v1=9',
            'brightness': 'v1=10',
            'contrast': 'v1=11',
            'saturation': 'v1=12',
            'entropy': 'v1=13',
            'walkability': 'v1=14',
            'objectness': 'v1=15',
            'semantic_diversity': 'v1=16',
            'geometric_complexity': 'v1=17',
            'occlusion_score': 'v1=18',
            'watermark_score': 'v1=19',
            'aesthetic_score': 'v1=20',
        }
        self.id_metadata_map = {v: k for k, v in self.metadata_id_map.items()}

        # Image-dimension modalities are binned into 32 bins
        self.image_dim_modalities = ['original_height', 'original_width']

        # Integer modalities that don't undergo any scaling (except for truncation)
        self.metadata_int_modalities = [
            'caption_n_chars', 'caption_n_words', 'caption_n_sentences', 
            'n_humans', 'n_sam_instances', 'n_coco_instances', 
            'coco_instance_diversity', 'semantic_diversity', 
        ]

        # Bin boundaries for manually defined metadata modalities.
        # Lowest and highest bin boundaries are implicitly set to -inf and +inf
        self.metadata_manual_bins = {
            'watermark_score': [0.5],
            'aesthetic_score': [4.5, 5.5],
        }

        # All other float or integer modalities that are binned into a defined number of bins
        # Dictionary entries are (vmin, vmax, num_bins)
        self.metadata_min_max_bins = {
            'colorfulness': (0, 150, 50),
            'brightness': (0, 255, 50),
            'contrast': (0, 127, 50),
            'saturation': (0, 255, 50),
            'entropy': (0, 10, 50),
            'walkability': (0, 1, 50),
            'objectness': (0, 1, 50),
            'geometric_complexity': (0, 0.75, 50),
            'occlusion_score': (0, 0.25, 50),
        }

    def image_dim_to_string(self, metadata, key, bin_size=32):
        value = metadata[key] // bin_size
        value = max(self.special_vmin, min(value, self.special_vmax))
        return f"{self.metadata_id_map[key]} v0={value}"

    def int_metadata_to_string(self, metadata, key):
        value = max(self.special_vmin, min(metadata[key], self.special_vmax))
        return f"{self.metadata_id_map[key]} v0={value}"

    def float_metadata_to_string(self, metadata, key, vmin, vmax, bins):
        value = max(vmin, min(metadata[key], vmax))
        value = (value - vmin) / (vmax - vmin)
        value = int(value * (bins-1))
        return f"{self.metadata_id_map[key]} v0={value}"
    
    def manual_bin_metadata_to_string(self, metadata, key):
        value = metadata[key]
        bin_idx = 0
        for bin_value in self.metadata_manual_bins[key]:
            if value < bin_value:
                break
            bin_idx += 1
        return f"{self.metadata_id_map[key]} v0={bin_idx}"
    
    def metadata_to_string(self, metadata, keys: List[str] = None):
        keys = list(metadata.keys()) if keys is None else keys

        if self.shuffle:
            # Randomly shuffle
            random.shuffle(keys)
        if self.random_trunc:
            # Randomly truncate
            keys = keys[:random.randint(1,len(keys))]

        metadata_strings = []
        
        for key in keys:
            if key in self.image_dim_modalities:
                # Image dimension modalities
                metadata_str = self.image_dim_to_string(metadata, key, bin_size=self.image_dim_bin_size)
            elif key in self.metadata_int_modalities:
                # Integer modalities that don't undergo any scaling
                metadata_str = self.int_metadata_to_string(metadata, key)
            elif key in self.metadata_manual_bins:
                # Metadata modalities for which bin boundaries are manually defined
                metadata_str = self.manual_bin_metadata_to_string(metadata, key)
            else:
                # All other modalities
                vmin, vmax, bins = self.metadata_min_max_bins[key]
                metadata_str = self.float_metadata_to_string(metadata, key, vmin, vmax, bins)

            metadata_strings.append(metadata_str)

        if self.return_chunks:
            return metadata_strings
        else:
            return ' '.join(metadata_strings)

    def load(self, path):
        with open(path, 'r') as f:
            sample = json.load(f)
        return sample

    def preprocess(self, sample):
        return sample

    def image_augment(self, val, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple, 
                      rand_aug_idx=None, resample_mode: str = None):
        return val

    def postprocess(self, metadata):
        if self.return_raw:
            return metadata
        metadata_str = self.metadata_to_string(metadata)
        return metadata_str
    

class HumanPoseTransform(AbstractTransform):

    def __init__(self, coord_bins=1000, only_pose=False, return_raw=False):
        self.coord_bins = coord_bins
        self.return_raw = return_raw
        self.only_pose = only_pose

    def convert_humanpose_instance(self, instances, only_pose=False):
        """Convert instances dict to list of lists where each list takes the form:
        [human, xmin xmax ymin ymax global val1 val2 ... val10 pose val1 val2 ... val 207 shape val1 val2 ... val10 camera val1 val2 val3 val4] 
        Like for bounding boxes, xmin, ymin, xmax, and ymax map to v0, v1, v2, and v3 respectively.
        """
        if only_pose: # used for tokenizer training for pose
            if len(instances) == 0:
                return torch.zeros(207)
            else:
                return torch.from_numpy(np.array(instances['pred_smpl_params']['body_pose'][0]).flatten()).float()
        if len(instances) == 0: #empty, i.e. there are no humans
            return 'none'
        
        for k in instances:
            if k!='pred_smpl_params':
                instances[k] = torch.from_numpy(np.array(instances[k]))

        smpl_params = (instances['pred_smpl_params'])

        for k in smpl_params:
            smpl_params[k] = torch.from_numpy(np.array(smpl_params[k]))

        total_num_instances = len(instances['bbox_xyxy'])
        instances_converted = []
        for ii in range(total_num_instances):
            instances_converted.append(['human'] + (np.array(instances['bbox_xyxy'][ii]).flatten().tolist()) + ['global'] + (np.array(instances['pred_smpl_params']['global_orient'][ii]).flatten().tolist()) + ['pose'] + (instances['pose_tokenized'][ii].flatten().tolist()) + ['shape'] + (instances['pred_smpl_params']['betas'][ii].flatten().tolist()) + ['camera'] + (instances['pred_cam'][ii].flatten().tolist()))
        return instances_converted

    def humanposes_crop_and_resize(self, humanposes: List[Tuple], crop_coords: Tuple, orig_size: Tuple,):
        """Crop and resize human poses (and their bounding boxes)
        """
        orig_height, orig_width = orig_size
        top, left, h, w = crop_coords

        humanposes_converted_resized = []
        for instance in humanposes:
            bbox_curr = instance[1:5]
            bbox_curr = np.array(bbox_curr)
            bbox_curr[0::2] = bbox_curr[0::2] / orig_width
            bbox_curr[1::2] = bbox_curr[1::2] / orig_height

            xmin, ymin, xmax, ymax = left, top, left + w, top + h
            bbox_curr = A.bbox_crop(bbox_curr, x_min=xmin, y_min=ymin, x_max=xmax, y_max=ymax, rows=orig_height,
                                        cols=orig_width) 
            bbox_curr = np.array(bbox_curr)
            if np.all(bbox_curr[1::2]<0) or np.all(bbox_curr[0::2]<0): #bbox is out of range, remove it 
                continue
            if np.all(bbox_curr[1::2]>1.0) or np.all(bbox_curr[0::2]>1.0): #bbox is out of range, remove it 
                continue
            bbox_curr = np.clip(bbox_curr, a_min=0, a_max=1.)

            instance[1:5] = bbox_curr
            humanposes_converted_resized.append(instance)

        # now return all instances, or none if there is no instance
        if len(humanposes_converted_resized)>0:
            pass
        else: #no valid masks remains
            return 'none'

        humanpose_returned = humanposes_converted_resized

        return humanpose_returned

    def convert_humanposes_to_string(self, all_humanposes: List[Tuple]):
        """Convert humanposes to a string
           range of global orientation: [-1, 1]
           range of object pose: [-1, 1]
           range of shape (betas): [-3, 3] 
           range of camera: [-1, 19]
        """
        bins = self.coord_bins

        instance_final_all = ''

        for humanposes in all_humanposes:
            human = humanposes[0]
            bboxes = humanposes[1:5]
            glob = humanposes[5]
            global_orient = np.array(humanposes[6:15])
            pose = humanposes[15]
            pose_params = np.array(humanposes[16:24]) 
            shape = humanposes[24] 
            shape_params = np.array(humanposes[25:35]) 
            camera = humanposes[35] 
            camera_params = np.clip(np.array(humanposes[36:]), a_min=-1., a_max=19.) 

            bboxes_new = [
                    f"v0={round(bboxes[0] * (bins - 1))}",
                    f"v1={round(bboxes[1] * (bins - 1))}",
                    f"v2={round(bboxes[2] * (bins - 1))}",
                    f"v3={round(bboxes[3] * (bins - 1))}"]

            global_orient = 499.5*global_orient
            global_orient_new = []
            for ii in range(len(global_orient)):
                global_orient_curr =  f"v0={round(global_orient[ii]+499.5)}"
                global_orient_new.append(global_orient_curr)

            pose_params_new = []
            for ii in range(len(pose_params)):
                if pose_params[ii]<512: 
                    pose_params_curr =  f"v0={round(pose_params[ii])}"
                else: 
                    pose_params_curr =  f"v1={round(pose_params[ii] - 512)}"
                pose_params_new.append(pose_params_curr)

            shape_params = 166.5*shape_params
            shape_params_new = []
            for ii in range(len(shape_params)):
                shape_params_curr =  f"v0={round(shape_params[ii]+499.5)}"
                shape_params_new.append(shape_params_curr)

            camera_params = 49.95*camera_params
            camera_params_new = []
            for ii in range(len(camera_params)):
                camera_params_curr =  f"v0={round(camera_params[ii]+49.95)}"
                camera_params_new.append(camera_params_curr)
            
            #randomly shuffle everything except bbox part of the sequence
            all_strings = [[pose]+pose_params_new, [glob] + global_orient_new, [camera] + camera_params_new, [shape] + shape_params_new ]
            rand_perm = torch.randperm(4)
            instance_final = [human] + bboxes_new + all_strings[rand_perm[0]] + all_strings[rand_perm[1]] + all_strings[rand_perm[2]] + all_strings[rand_perm[3]]
            
        
            instance_final = ', '.join(instance_final)
            instance_final = instance_final.replace(",", "")
            instance_final_all = instance_final_all + instance_final + ' '

        return instance_final_all 

    def load(self, path):
        with open(path, 'r') as f:
            sample = json.load(f)

        return sample

    def preprocess(self, sample):
        instances = sample 
        instances = self.convert_humanpose_instance(instances, only_pose=self.only_pose)
        return instances

    def image_augment(self, humanposes: List[Tuple], crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple, 
                      rand_aug_idx=None, resample_mode: str = None):
        if humanposes=='none' or self.only_pose:
            return humanposes
        humanposes = self.humanposes_crop_and_resize(humanposes, crop_coords, orig_size)
        return humanposes

    def postprocess(self, humanposes):
        if humanposes=='none' or self.only_pose:
            return humanposes if not self.return_raw else []
        if self.return_raw:
            return humanposes
        humanposes = self.convert_humanposes_to_string(humanposes)
        return humanposes


class ColorPaletteTransform(AbstractTransform):

    def __init__(self, coord_bins=1000, return_raw=False):
        self.coord_bins = coord_bins
        self.return_raw = return_raw

    def convert_palette_instance(self, instances):
        """Convert colors to v0= v0= ...
        """
        length = random.randint(1,7)
        instances_converted = np.array(instances[0][str(length)]).flatten().tolist()
        return instances_converted

    def palette_hflip(self, palettes: List[Tuple], image_size: Tuple, flip: bool):

        return palettes

    def convert_palettes_to_string(self, all_palettes: List[Tuple]):
        """Convert palettes to a string
        """

        colors = []
        len_palettes = len(all_palettes)
        colors.append(f"v1={round(len_palettes/3)}") # start with the length of the color palette to avoid confusion
        for ii in range(len(all_palettes)):
            color_new = f"v0={round(all_palettes[ii])}"
            colors.append(color_new)
        
        instance_final_all = colors
        instance_final_all = ', '.join(instance_final_all)
        instance_final_all = instance_final_all.replace(",", "")

        return instance_final_all 

    def load(self, path):
        with open(path, 'r') as f:
            sample = json.load(f)
        return sample

    def preprocess(self, sample):
        if self.return_raw:
            return sample
        instances = sample 
        instances = self.convert_palette_instance(instances)
        return instances

    def image_augment(self, palettes: List[Tuple], crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple, 
                      rand_aug_idx=None, resample_mode: str = None):
        return palettes

    def postprocess(self, palettes):
        if self.return_raw:
            return palettes
        palettes = self.convert_palettes_to_string(palettes)
        return palettes
    

class SAMInstanceTokTransform(AbstractTransform):

    def __init__(self, image_size=224, points_per_side=7, point_order='random'):
        self.H, self.W = to_2tuple(image_size)
        self.points_per_h, self.points_per_w = to_2tuple(points_per_side)
        assert point_order in ['random', 'grid']
        self.point_order = point_order

    def get_query_points(self):
        if self.point_order == 'grid':
            # Create and cache grid query points
            if not hasattr(self, 'grid_query_points'):
                y, x = np.meshgrid(np.linspace(0, self.H, self.points_per_h + 2)[1:-1], np.linspace(0, self.W, self.points_per_w + 2)[1:-1])
                grid = np.stack((x, y), axis=2).astype(np.int32)
                self.grid_query_points = grid.reshape(-1, 2)
            return self.grid_query_points
        elif self.point_order == 'random':
            # Randomly sample query points
            y = np.random.randint(0, self.H, self.points_per_h)
            x = np.random.randint(0, self.W, self.points_per_w)
            return np.concatenate((x[:,None], y[:,None]), axis=1)
        else:
            raise ValueError(f"Query point order mode {self.point_order} is not supported.")

    def get_target_tokens(self, sample, query_points):
        instances_coords = [coords[0] for coords in sample['points']]
        tokens = sample['token_ids']
        bboxes = sample['bbox']
        
        instance_tokens_per_qpoint = dict()
        for point in query_points:
            point = (int(point[0].item()), int(point[1].item()))
            instance_tokens_per_qpoint[point] = []
            for i, (coords, tok, bbox) in enumerate(zip(instances_coords, tokens, bboxes)):
                # Calculate the distance from the query point to the instance
                distance = cv2.pointPolygonTest(coords, point, measureDist=True)
                # If the query point is inside the instance, add its corresponding token
                if distance >= 0:
                    instance_tokens_per_qpoint[point].append((tok, bbox))
        
        return instance_tokens_per_qpoint

    def convert_target_tokens_to_string(self, target_tokens):
        result_text = []
        query_points = list(target_tokens.keys())
        # Randomly shuffle query points order (mainly for grid order)
        random.shuffle(query_points)
        for point in query_points:
            
            # Add query point coordinates to the string
            result_text.append('point')
            result_text.append(f'v0={point[1]}')
            result_text.append(f'v1={point[0]}')
            
            # Randomly shuffle the order of instance tokens per query point
            random.shuffle(target_tokens[point])
            if len(target_tokens[point]) == 0:
                # If no instances tokens are found, add 'none' to the string
                result_text.append('none')
            else:
                for tok, bbox in target_tokens[point]:
                    result_text.append(f'polygon')
                    
                    # Add bounding box coordinates to the string
                    ymin, xmin, ymax, xmax = bbox.astype(np.int32)
                    result_text.extend([
                        f'v0={xmin}',
                        f'v1={ymin}',
                        f'v2={xmax}',
                        f'v3={ymax}',
                    ])
                    
                    # Add instance tokens ids to the string
                    for idx in tok.tolist():
                        if idx < 512:
                            result_text.append(f'v0={idx}')
                        else:
                            result_text.append(f'v1={idx - 512}')
        
        return " ".join(result_text)

    def load(self, path):
        sample = np.load(path, allow_pickle=True)
        return sample

    def preprocess(self, sample):
        for s in sample:
            s['token_ids'] = s['token_ids'].astype(np.int32)
        return sample

    def image_augment(self, v, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        if rand_aug_idx is None:
            raise ValueError("Crop settings / augmentation index are missing but a pre-tokenized modality is being used")
        v = v[rand_aug_idx]
        return v

    def postprocess(self, sample):
        query_points = self.get_query_points()
        target_tokens = self.get_target_tokens(sample, query_points)
        final_string = self.convert_target_tokens_to_string(target_tokens)
        return final_string


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