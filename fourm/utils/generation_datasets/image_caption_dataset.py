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
import os
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from fourm.data.multimodal_dataset_folder import make_dataset, UNIFIED_EXTENSIONS
from fourm.data.modality_transforms import get_transform_key, RGBTransform, CaptionTransform, UnifiedDataTransform


class ImageCaptionDataset(Dataset):
    """
    Similar to MultiModalDatasetFolder, but specialized for image-caption datasets.
    """
    def __init__(self, 
                 root: str, 
                 augmenter: Optional[Callable] = None,
                 modality_paths: Dict[str, str] = None, 
                 is_valid_file: Optional[Callable[[str], bool]] = None, 
                 cache=False):
        self.root = root
        self.modality_paths = modality_paths or {}
        
        self.modality_transforms = {
            'rgb': RGBTransform(imagenet_default_mean_and_std=False),
            'caption': CaptionTransform()
        }
        
        self.transform = UnifiedDataTransform(transforms_dict=self.modality_transforms, image_augmenter=augmenter)
        
        classes, class_to_idx = self._find_classes(os.path.join(self.root, self.modality_paths.get('caption', 'caption')))
        extensions = UNIFIED_EXTENSIONS if is_valid_file is None else None
        
        samples = {
            mod: make_dataset(
                os.path.join(self.root, self.modality_paths.get(mod, mod)),
                class_to_idx, 
                extensions, 
                is_valid_file,
                cache_path=os.path.join(self.root, 'dataloader_cache', f'{self.modality_paths.get(mod, mod)}.pkl') if cache else None)
            for mod in ['caption', 'rgb']
        }
        
        for mod, mod_samples in samples.items():
            if len(mod_samples) == 0:
                msg = "Found 0 logs in subfolders of: {}\n".format(os.path.join(self.root, self.modality_paths.get(mod, mod)))
                if extensions is not None:
                    msg += "Supported extensions are: {}".format(",".join(extensions))
                raise RuntimeError(msg)

        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        
    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
        
    def __getitem__(self, index):
        
        sample_dict = {}
        for mod in ['caption', 'rgb']:
            path, _ = self.samples[mod][index]
            sample = self.modality_transforms[get_transform_key(mod)].load(path)
            sample_dict[mod] = sample
            
        if self.transform is not None:
            sample_dict = self.transform(sample_dict)
        
        return sample_dict
    
    def __len__(self) -> int:
        return len(list(self.samples.values())[0])