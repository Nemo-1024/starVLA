from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch
from typing_extensions import NotRequired, Required, TypedDict

TensorLike2D = Union[torch.Tensor, np.ndarray, Sequence[float], Sequence[Sequence[float]]]
FrameArray = np.ndarray
ImageViews = Sequence[FrameArray]
VideoViews = Sequence[Sequence[FrameArray]]


class LiberoExample(TypedDict, total=False):
    image: Required[ImageViews]
    lang: Required[str]
    state: Required[TensorLike2D]
    embodiment_id: Required[int]
    action: NotRequired[TensorLike2D]
    primary_videos: NotRequired[VideoViews]
    video: NotRequired[VideoViews]
    wrist_images: NotRequired[ImageViews]


class LatentWorldPolicyInferBatch(TypedDict):
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    act_placeholder_mask: torch.Tensor
    flow_placeholder_mask: torch.Tensor
    lam_videos: torch.Tensor
    state: torch.Tensor
    state_mask: torch.Tensor
    embodiment_id: torch.Tensor
    image_grid_thw: Optional[torch.Tensor]
    wrist_videos: Optional[torch.Tensor]


class LatentWorldPolicyTrainBatch(LatentWorldPolicyInferBatch):
    actions: torch.Tensor
    actions_mask: torch.Tensor
