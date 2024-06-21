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
import numpy as np
from torch.utils.data import Dataset


class RepeatedDatasetWrapper(Dataset):
    def __init__(self, original_dataset, num_repeats):
        """
        Dataset wrapper that repeats the original dataset n times.

        Args:
            original_dataset (torch.utils.data.Dataset): The original dataset to be repeated.
            num_repeats (int): The number of times the dataset should be repeated.
        """
        self.original_dataset = original_dataset
        self.num_repeats = num_repeats

    def __getitem__(self, index):
        """
        Retrieve the item at the given index.
        
        Args:
            index (int): The index of the item to be retrieved.
        """
        original_index = index % len(self.original_dataset)
        return self.original_dataset[original_index]

    def __len__(self):
        """
        Get the length of the dataset after repeating it n times.
        
        Returns:
            int: The length of the dataset.
        """
        return len(self.original_dataset) * self.num_repeats


class SubsampleDatasetWrapper(Dataset):
    def __init__(self, original_dataset, dataset_size, seed=0, return_orig_idx=False):
        """
        Dataset wrapper that randomly subsamples the original dataset.

        Args:
            original_dataset (torch.utils.data.Dataset): The original dataset to be subsampled.
            dataset_size (int): The size of the subsampled dataset.
            seed (int): The seed to use for selecting the subset of indices of the original dataset.
            return_orig_idx (bool): Whether to return the original index of the item in the original dataset.
        """
        self.original_dataset = original_dataset
        self.dataset_size = dataset_size or len(original_dataset)
        self.return_orig_idx = return_orig_idx
        np.random.seed(seed)
        self.indices = np.random.permutation(len(self.original_dataset))[:self.dataset_size]

    def __getitem__(self, index):
        """
        Retrieve the item at the given index.
        
        Args:
            index (int): The index of the item to be retrieved.
        """
        original_index = self.indices[index]
        sample = self.original_dataset[original_index]
        return sample, original_index if self.return_orig_idx else sample

    def __len__(self):
        """
        Get the length of the dataset after subsampling it.
        
        Returns:
            int: The length of the dataset.
        """
        return len(self.indices)
