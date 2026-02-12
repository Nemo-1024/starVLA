from typing import Dict, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import RandomResizedCrop


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_IMAGENET_STATS_CACHE: Dict[Tuple[torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}

# 宽高比 ratio = width/height。640×480 → 4:3 ≈ 1.333
ASPECT_4_3 = 4.0 / 3.0


def _numpy_frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """Convert uint8 NHWC -> float tensor [T,3,H,W] in [0,1]."""
    assert frames.ndim == 4, f"Expected (T,H,W,C), got {frames.shape}"
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    return tensor


def _apply_crop(frames: torch.Tensor, params: Tuple[int, int, int, int], size: Tuple[int, int]) -> torch.Tensor:
    """Apply the same crop to all frames, then resize to target size."""
    i, j, h, w = params
    cropped = F.resized_crop(frames, i, j, h, w, size=size, antialias=True)
    return cropped


def two_view_video_aug(
    frames_uint8: np.ndarray,
    output_size: Tuple[int, int] = (256, 256),  # 修改为256x256以匹配grid_size=16
    scale: Tuple[float, float] = (0.8, 1.0),
    # ratio = (宽高比下界, 上界)。640×480 用 (1.0, 4/3)，允许从正方形到 4:3
    ratio: Tuple[float, float] = (1.0, ASPECT_4_3),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate two augmented views from the same clip.

    Returns:
        videos, dec_videos: both [T,3,H,W] float in [0,1]
    """
    frames = _numpy_frames_to_tensor(frames_uint8)
    # view 1
    params1 = RandomResizedCrop.get_params(frames[0], scale=scale, ratio=ratio)
    video1 = _apply_crop(frames, params1, output_size)
    # view 2
    params2 = RandomResizedCrop.get_params(frames[0], scale=scale, ratio=ratio)
    video2 = _apply_crop(frames, params2, output_size)
    return video1, video2


def _sample_jitter_factors(
    device: torch.device,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
    generator: torch.Generator | None = None,
) -> Tuple[float, float, float, float]:
    def _uniform(low: float, high: float) -> float:
        if generator is None:
            return float(torch.empty((), device=device).uniform_(low, high).item())
        return float((low + (high - low) * torch.rand((), generator=generator)).item())

    b = _uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
    c = _uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
    s = _uniform(max(0.0, 1.0 - saturation), 1.0 + saturation)
    h = _uniform(-hue, hue)
    return b, c, s, h


def _sample_jitter_order(generator: torch.Generator | None = None) -> list[int]:
    if generator is None:
        return torch.randperm(4).tolist()
    return torch.randperm(4, generator=generator).tolist()


def _apply_color_jitter(
    frames: torch.Tensor,
    brightness_factor: float,
    contrast_factor: float,
    saturation_factor: float,
    hue_factor: float,
    order: list[int],
) -> torch.Tensor:
    out = frames
    for op in order:
        if op == 0:
            out = F.adjust_brightness(out, brightness_factor)
        elif op == 1:
            out = F.adjust_contrast(out, contrast_factor)
        elif op == 2:
            out = F.adjust_saturation(out, saturation_factor)
        else:
            out = F.adjust_hue(out, hue_factor)
        out = out.clamp_(0.0, 1.0)
    return out


def _deterministic_resize_center_crop(frames: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    short_side = min(output_size)
    resized = F.resize(frames, short_side, antialias=True)
    return F.center_crop(resized, output_size)


def _to_batched_chw_float(videos_uint8: torch.Tensor) -> torch.Tensor:
    if videos_uint8.ndim != 5:
        raise ValueError(f"Expected 5D tensor, got {tuple(videos_uint8.shape)}")
    if videos_uint8.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 videos, got {videos_uint8.dtype}")

    if videos_uint8.shape[-1] == 3:
        frames = videos_uint8.permute(0, 1, 4, 2, 3)
    elif videos_uint8.shape[2] == 3:
        frames = videos_uint8
    else:
        raise ValueError(f"Expected channel dim=3 in axis 2 or -1, got {tuple(videos_uint8.shape)}")

    return frames.to(dtype=torch.float32).mul_(1.0 / 255.0)


def gpu_two_view_video_aug(
    videos_uint8: torch.Tensor,
    *,
    output_size: Tuple[int, int] = (256, 256),
    scale: Tuple[float, float] = (0.8, 1.0),
    ratio: Tuple[float, float] = (1.0, ASPECT_4_3),
    brightness: float = 0.3,
    contrast: float = 0.4,
    saturation: float = 0.5,
    hue: float = 0.08,
    training: bool = True,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-side two-view video augmentation.

    Args:
        videos_uint8: [B,T,H,W,C] or [B,T,C,H,W] uint8 tensor.
    Returns:
        video1, video2: [B,T,3,OH,OW] float32, ImageNet normalized.
    """
    frames = _to_batched_chw_float(videos_uint8)
    batch_size, timesteps = int(frames.shape[0]), int(frames.shape[1])
    out_h, out_w = output_size
    video1 = torch.empty((batch_size, timesteps, 3, out_h, out_w), device=frames.device, dtype=torch.float32)
    video2 = torch.empty_like(video1)

    for idx in range(batch_size):
        clip = frames[idx]  # [T,3,H,W]

        if training:
            params1 = RandomResizedCrop.get_params(clip[0], scale=scale, ratio=ratio)
            aug1 = _apply_crop(clip, params1, output_size)
            b1, c1, s1, h1 = _sample_jitter_factors(
                clip.device, brightness, contrast, saturation, hue, generator=generator
            )
            aug1 = _apply_color_jitter(aug1, b1, c1, s1, h1, _sample_jitter_order(generator=generator))

            params2 = RandomResizedCrop.get_params(clip[0], scale=scale, ratio=ratio)
            aug2 = _apply_crop(clip, params2, output_size)
            b2, c2, s2, h2 = _sample_jitter_factors(
                clip.device, brightness, contrast, saturation, hue, generator=generator
            )
            aug2 = _apply_color_jitter(aug2, b2, c2, s2, h2, _sample_jitter_order(generator=generator))
        else:
            aug1 = _deterministic_resize_center_crop(clip, output_size)
            aug2 = aug1

        video1[idx] = aug1
        video2[idx] = aug2

    imagenet_normalize_(video1)
    imagenet_normalize_(video2)
    return video1, video2


def _get_imagenet_stats(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (device, dtype)
    if key not in _IMAGENET_STATS_CACHE:
        mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
        _IMAGENET_STATS_CACHE[key] = (mean, std)
    return _IMAGENET_STATS_CACHE[key]


def imagenet_normalize_(tensor: torch.Tensor) -> torch.Tensor:
    """
    In-place ImageNet normalization for tensor [...,3,H,W].
    """
    mean, std = _get_imagenet_stats(tensor.device, tensor.dtype)
    tensor.sub_(mean).div_(std)
    return tensor
