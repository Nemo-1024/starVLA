from typing import Any, Dict, List, Sequence

import torch


def _as_video_uint8_nhwc(frames: Any) -> torch.Tensor:
    if isinstance(frames, torch.Tensor):
        video = frames
    else:
        video = torch.as_tensor(frames)

    if video.ndim != 4:
        raise ValueError(f"Expected frames with 4 dims, got shape={tuple(video.shape)}")

    if video.shape[-1] == 3:
        video_nhwc = video
    elif video.shape[1] == 3:
        video_nhwc = video.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Expected channel dim=3 in axis 1 or -1, got shape={tuple(video.shape)}")

    if video_nhwc.dtype != torch.uint8:
        if torch.is_floating_point(video_nhwc):
            max_val = float(video_nhwc.max().item()) if video_nhwc.numel() > 0 else 0.0
            scale = 255.0 if max_val <= 1.0 + 1e-6 else 1.0
            video_nhwc = video_nhwc.mul(scale).clamp(0, 255).to(torch.uint8)
        else:
            video_nhwc = video_nhwc.clamp(0, 255).to(torch.uint8)

    return video_nhwc.contiguous()


def _as_proprio_float32(proprio: Any) -> torch.Tensor:
    if isinstance(proprio, torch.Tensor):
        proprio_t = proprio
    else:
        proprio_t = torch.as_tensor(proprio)
    if proprio_t.ndim != 2:
        raise ValueError(f"Expected proprio with shape [T,D], got {tuple(proprio_t.shape)}")
    return proprio_t.to(torch.float32)


def lam_collate(batch: Sequence[Dict], max_proprio_dim: int = 32) -> Dict[str, Any]:
    """
    Collate raw clips and proprio into a batch.

    Expects each item: {"frames": np[T,H,W,C], "proprio": np[T,D], "dataset_id": int}

    Returns:
        videos: either stacked uint8 [B,T,H,W,C] (homogeneous shapes) or list of uint8 [T,H,W,C]
        proprio: [B,T,max_proprio_dim] float32
        delta_proprio: [B,max_proprio_dim] float32
        dataset_ids: [B] int64
        proprio_mask: [B,max_proprio_dim] float32
    """
    if len(batch) == 0:
        raise ValueError("lam_collate received an empty batch.")

    batch_size = len(batch)
    first_proprio = _as_proprio_float32(batch[0]["proprio"])
    time_steps = int(first_proprio.shape[0])

    proprio_t = torch.zeros((batch_size, time_steps, max_proprio_dim), dtype=torch.float32)
    delta_t = torch.zeros((batch_size, max_proprio_dim), dtype=torch.float32)
    proprio_mask_t = torch.zeros((batch_size, max_proprio_dim), dtype=torch.float32)
    dataset_ids_t = torch.empty((batch_size,), dtype=torch.long)

    frames_t_list: List[torch.Tensor] = []
    first_video_shape = None
    homogeneous_shape = True

    for i, sample in enumerate(batch):
        frames_t = _as_video_uint8_nhwc(sample["frames"])
        if frames_t.shape[0] != time_steps:
            raise ValueError(
                f"Inconsistent temporal length in batch: expected video T={time_steps}, got T={frames_t.shape[0]}"
            )
        if first_video_shape is None:
            first_video_shape = tuple(frames_t.shape)
        elif tuple(frames_t.shape) != first_video_shape:
            homogeneous_shape = False
        frames_t_list.append(frames_t)

        proprio = _as_proprio_float32(sample["proprio"])
        if proprio.shape[0] != time_steps:
            raise ValueError(
                f"Inconsistent temporal length in batch: expected proprio T={time_steps}, got T={proprio.shape[0]}"
            )
        original_dim = int(proprio.shape[-1])
        if original_dim > max_proprio_dim:
            raise ValueError(
                f"Sample proprio dim {original_dim} exceeds max_proprio_dim {max_proprio_dim}. "
                f"Please increase max_proprio_dim in config."
            )

        proprio_t[i, :, :original_dim] = proprio
        delta_t[i, :original_dim] = proprio[-1, :original_dim] - proprio[0, :original_dim]
        proprio_mask_t[i, :original_dim] = 1.0
        dataset_ids_t[i] = int(sample.get("dataset_id", 0))

    if homogeneous_shape:
        videos_out: torch.Tensor | List[torch.Tensor] = torch.stack(frames_t_list, dim=0)
    else:
        videos_out = frames_t_list

    return {
        "videos": videos_out,
        "proprio": proprio_t,
        "delta_proprio": delta_t,
        "dataset_ids": dataset_ids_t,
        "proprio_mask": proprio_mask_t,
    }
