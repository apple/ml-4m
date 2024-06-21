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
from torch.utils.data import Dataset

class EmptyDataset(Dataset):
    """Empty dataset"""

    def __init__(self, dataset_size: int):
        self.dataset_size = dataset_size
        
    def __getitem__(self, index):
        return {}
    
    def __len__(self):
        return self.dataset_size