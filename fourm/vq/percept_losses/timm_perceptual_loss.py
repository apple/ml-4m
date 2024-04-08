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
from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.models.feature_extraction import create_feature_extractor
from einops import rearrange
import timm


class TimmPerceptualLoss(nn.Module):
    """Perceptual loss module using features from arbitrary timm models.

    Args:
        model_id: timm model id. E.g. 'vit_base_patch14_dinov2.lvd142m'
        feature_ids: List or hyphen-separated string of feature names to extract from the model. 
          For example, 'blocks.2-blocks.5-blocks.8-blocks.11'. To list all available features, use:
          ```python
          from torchvision.models.feature_extraction import get_graph_node_names
          nodes, _ = get_graph_node_names(model)
          ```
        feature_loss: Feature loss to use. One of ['cosine' or 'cos', 'l1' or 'mae']. Default: 'cosine'.
          If 'l1' or 'mae' is used, the features will be normalized first. If 'cosine' or 'cos' is 
          used, the features will not be normalized, but the cosine similarity will be computed, 
          which is equivalent to normalization + MSE up to a factor of 2.
    """
    def __init__(self, 
                 model_id: str, 
                 feature_ids: Union[str, List[str]],
                 feature_loss: str = 'cosine'):
        super().__init__()

        feature_ids = feature_ids.split('-') if isinstance(feature_ids, str) else feature_ids

        self.feature_ids = feature_ids
        self.feature_loss = feature_loss

        self.model = timm.create_model(model_id, pretrained=True)
        self.feature_extractor = create_feature_extractor(self.model, return_nodes=self.feature_ids)

        # Transforms to preprocess inputs to the model
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.percept_transform = transforms.Compose([
            transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)), # [-1, 1] -> [0, 1]
            transforms.Normalize(self.data_config['mean'], self.data_config['std']), # [0, 1] -> standardize with pre-computed statistics
            transforms.Resize(self.data_config['input_size'][-2:], interpolation=TF.InterpolationMode.BILINEAR, antialias=True),
        ])

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, preprocess_inputs=False) -> torch.Tensor:
        """
        Compute perceptual loss between predictions and targets. If 
        preprocess_inputs is True, it is assumed that the targets are 
        scaled to the [-1, 1] range. Predictions will be scaled 
        assuming the same input range.
        
        Args:
            preds: Predictions tensor of shape (B, C, H, W)
            targets: Targets tensor of shape (B, C, H, W)
            preprocess_inputs: If inputs are scaled to [-1, 1], enable 
              this to apply model specific preprocessing. Default: False.

        Returns:
            Perceptual loss between predictions and targets.
        """
        # Preprocess predictions and targets for the given feature extractor
        if preprocess_inputs:
            preds = self.percept_transform(preds)
            targets = self.percept_transform(targets)

        # Extract features from predictions and targets
        # Each is a dict of feature_name: feature_tensor
        feats_preds = self.feature_extractor(preds)
        feats_targets = self.feature_extractor(targets)

        loss = 0
        for feat_name in feats_preds.keys():

            # Get individual feature map and reshape from (B, C, H, W) to (B, N, C) if needed
            feat_preds = feats_preds[feat_name]
            feat_targets = feats_targets[feat_name]
            if feat_preds.ndim == 4:
                feat_preds = rearrange(feat_preds, 'b c h w -> b (h w) c')
                feat_targets = rearrange(feat_targets, 'b c h w -> b (h w) c')
            
            # Compute feature-wise loss
            if self.feature_loss in ['l1', 'mae']:
                feat_preds = F.normalize(feat_preds, dim=-1)
                feat_targets = F.normalize(feat_targets, dim=-1)
                loss += F.l1_loss(feat_preds, feat_targets, reduction='none').sum(-1).mean(-1)
            elif self.feature_loss in ['cosine', 'cos']:
                loss += 1 - F.cosine_similarity(feat_preds, feat_targets, dim=-1).mean(dim=-1)
            else:
                raise ValueError(f'Unknown feature loss: {self.feature_loss}')
        
        loss /= preds.shape[0]

        return loss