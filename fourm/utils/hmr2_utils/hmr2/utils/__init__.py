# --------------------------------------------------------
# Based on the 4DHumans and ProHMR code bases
# https://github.com/shubham-goel/4D-Humans
# https://github.com/nkolot/ProHMR
# --------------------------------------------------------

import torch
from typing import Any

from .renderer import Renderer
from .mesh_renderer import MeshRenderer
from .skeleton_renderer import SkeletonRenderer
# from .pose_utils import eval_pose, Evaluator

def recursive_to(x: Any, target: torch.device):
    """
    Recursively transfer a batch of data to the target device
    Args:
        x (Any): Batch of data.
        target (torch.device): Target device.
    Returns:
        Batch of data where all tensors are transfered to the target device.
    """
    if isinstance(x, dict):
        return {k: recursive_to(v, target) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(target)
    elif isinstance(x, list):
        return [recursive_to(i, target) for i in x]
    else:
        return x
