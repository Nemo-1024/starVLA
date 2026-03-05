from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from .types import (
    LatentWorldPolicyInferBatch,
    LatentWorldPolicyInferExample,
    LatentWorldPolicyTrainBatch,
    LatentWorldPolicyTrainExample,
)

MAX_SOURCE_ASPECT = 4.0 / 3.0


def _preprocess_numpy_frame_to_pil(frame: np.ndarray, target_hw: Tuple[int, int]) -> Image.Image:
    frame_arr = frame
    if np.issubdtype(frame_arr.dtype, np.floating):
        max_val = float(np.max(frame_arr)) if frame_arr.size > 0 else 0.0
        if max_val <= 1.0 + 1e-6:
            frame_arr = frame_arr * 255.0
    frame_u8 = np.clip(frame_arr, 0, 255).astype(np.uint8)

    if frame_u8.ndim == 3 and frame_u8.shape[2] == 1:
        frame_u8 = frame_u8[:, :, 0]

    image = Image.fromarray(frame_u8).convert("RGB")
    width, height = image.size

    if (width / height) > MAX_SOURCE_ASPECT:
        crop_width = max(1, int(np.floor(height * MAX_SOURCE_ASPECT)))
        left = (width - crop_width) // 2
        image = image.crop((left, 0, left + crop_width, height))

    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    return image.resize((target_w, target_h), resample=Image.BILINEAR)


def _numpy_video_to_tchw_float(video_frames: Sequence[np.ndarray], target_hw: Tuple[int, int]) -> torch.Tensor:
    if len(video_frames) == 0:
        raise ValueError("video_frames must be non-empty.")
    frame_arrays = []
    for frame in video_frames:
        pil_frame = _preprocess_numpy_frame_to_pil(frame, target_hw=target_hw)
        frame_arrays.append(torch.from_numpy(np.asarray(pil_frame)))
    video_thwc = torch.stack(frame_arrays, dim=0)  # [T, H, W, C], uint8
    return video_thwc.permute(0, 3, 1, 2).to(dtype=torch.float32).div_(255.0)


def _numpy_image_to_chw_float(frame: np.ndarray, target_hw: Tuple[int, int]) -> torch.Tensor:
    pil_frame = _preprocess_numpy_frame_to_pil(frame, target_hw=target_hw)
    return torch.from_numpy(np.asarray(pil_frame)).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)


class _LatentWorldPolicyBatchBuilderBase:
    def __init__(
        self,
        *,
        policy_cfg,
        policy_backend,
        policy_vlm_adapter,
        lam_image_hw: Tuple[int, int],
    ) -> None:
        self.policy_cfg = policy_cfg
        self.policy_backend = policy_backend
        self.policy_vlm_adapter = policy_vlm_adapter
        self.lam_image_hw = (int(lam_image_hw[0]), int(lam_image_hw[1]))

    @staticmethod
    def _formalize_language(language: str) -> str:
        language = language.lower()
        language = re.sub(r"[^\w\s]", "", language)
        return language

    @staticmethod
    def _align_feature_dim_with_mask(seq: torch.Tensor, target_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq.ndim != 2:
            raise ValueError(f"Expected [T, D] tensor, got shape={tuple(seq.shape)}")
        if target_dim < 0:
            raise ValueError(f"target_dim must be >= 0, got {target_dim}")

        curr_dim = int(seq.shape[1])
        if target_dim == 0:
            aligned = seq[:, :0]
            mask = torch.zeros((0,), dtype=torch.bool)
            return aligned, mask

        if curr_dim == target_dim:
            return seq, torch.ones((target_dim,), dtype=torch.bool)

        if curr_dim > target_dim:
            return seq[:, :target_dim], torch.ones((target_dim,), dtype=torch.bool)

        pad = torch.zeros(
            (int(seq.shape[0]), target_dim - curr_dim),
            dtype=seq.dtype,
            device=seq.device,
        )
        aligned = torch.cat([seq, pad], dim=1)
        mask = torch.zeros((target_dim,), dtype=torch.bool)
        mask[:curr_dim] = True
        return aligned, mask

    @staticmethod
    def _infer_binary_gripper_dims(seq: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 2:
            raise ValueError(f"Expected [T, D] tensor, got shape={tuple(seq.shape)}")
        action_dim = int(seq.shape[1])
        gripper_mask = torch.zeros((action_dim,), dtype=torch.bool, device=seq.device)
        if action_dim == 7:
            gripper_mask[6] = True
            return gripper_mask
        if action_dim == 14:
            gripper_mask[6] = True
            gripper_mask[13] = True
            return gripper_mask
        raise ValueError(
            "Unsupported action feature dimension for gripper index inference. "
            f"Expected D in {{7, 14}}, got D={action_dim}."
        )

    @staticmethod
    def _downsample_action_by_factor2(seq: torch.Tensor, gripper_mask: torch.Tensor) -> torch.Tensor:
        if seq.ndim != 2:
            raise ValueError(f"Expected [T, D] tensor, got shape={tuple(seq.shape)}")
        if gripper_mask.ndim != 1 or int(gripper_mask.shape[0]) != int(seq.shape[1]):
            raise ValueError(
                f"`gripper_mask` must be [D], got shape={tuple(gripper_mask.shape)} for D={int(seq.shape[1])}."
            )

        pair_count = int(seq.shape[0] // 2)
        if pair_count == 0:
            return seq[:0]

        even_steps = seq[: pair_count * 2 : 2]
        odd_steps = seq[1 : pair_count * 2 : 2]
        downsampled = (even_steps + odd_steps) * 0.5

        gripper_mask = gripper_mask.to(device=seq.device, dtype=torch.bool)
        if bool(gripper_mask.any()):
            downsampled = downsampled.clone()
            downsampled[:, gripper_mask] = odd_steps[:, gripper_mask]
        return downsampled

    @staticmethod
    def _sample_or_pad_sequence_with_mask(
        seq: torch.Tensor, target_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        if seq.shape[0] == target_len:
            return seq, torch.ones((target_len,), dtype=torch.bool), False
        was_downsampled = False
        if seq.shape[0] > target_len:
            original_len = int(seq.shape[0])
            gripper_mask = _LatentWorldPolicyBatchBuilderBase._infer_binary_gripper_dims(seq)
            seq = _LatentWorldPolicyBatchBuilderBase._downsample_action_by_factor2(seq, gripper_mask)
            was_downsampled = True
            downsampled_len = int(seq.shape[0])
            if downsampled_len > target_len:
                raise ValueError(
                    "Action sequence still exceeds configured action_horizon after fixed 2x downsampling. "
                    f"original_len={original_len}, downsampled_len={downsampled_len}, target_len={int(target_len)}. "
                    "Please increase `framework.action_model.action_horizon` or reduce data action window."
                )
            if downsampled_len == target_len:
                return seq, torch.ones((target_len,), dtype=torch.bool), True
        pad_len = target_len - seq.shape[0]
        pad = seq[-1:].repeat(pad_len, 1)
        out = torch.cat([seq, pad], dim=0)
        mask = torch.zeros((target_len,), dtype=torch.bool)
        mask[: seq.shape[0]] = True
        return out, mask, was_downsampled

    @staticmethod
    def _build_masks(
        input_ids: torch.Tensor,
        *,
        act_queries: int,
        flow_queries: int,
        placeholder_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        expected = int(act_queries + flow_queries)
        placeholder = input_ids == int(placeholder_id)
        order = placeholder.cumsum(dim=1)
        act_mask = placeholder & (order <= int(act_queries))
        flow_mask = placeholder & (order > int(act_queries)) & (order <= expected)
        return act_mask, flow_mask

    @staticmethod
    def _to_cpu_float_tensor(value, *, field_name: str, ex_idx: int) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value.detach()
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        elif isinstance(value, (list, tuple)):
            tensor = torch.as_tensor(value)
        else:
            raise TypeError(
                f"examples[{ex_idx}]['{field_name}'] must be torch.Tensor/np.ndarray/list/tuple, "
                f"got type={type(value)}."
            )
        return tensor.to(device="cpu", dtype=torch.float32)

    @classmethod
    def _to_2d_sequence_tensor(cls, value, *, field_name: str, ex_idx: int) -> torch.Tensor:
        tensor = cls._to_cpu_float_tensor(value, field_name=field_name, ex_idx=ex_idx)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2:
            raise ValueError(
                f"examples[{ex_idx}]['{field_name}'] must have shape [D] or [T, D], got {tuple(tensor.shape)}."
            )
        return tensor

    @staticmethod
    def _extract_primary_frame(primary_image, *, ex_idx: int) -> np.ndarray:
        if not isinstance(primary_image, (list, tuple)):
            raise ValueError(
                f"examples[{ex_idx}]['primary_image'] must be a list/tuple, got type={type(primary_image)}."
            )
        if len(primary_image) != 1:
            raise ValueError(
                f"examples[{ex_idx}]['primary_image'] must contain exactly 1 primary view, got {len(primary_image)}."
            )
        primary_frame = primary_image[0]
        if not isinstance(primary_frame, np.ndarray):
            raise ValueError(
                f"examples[{ex_idx}]['primary_image'][0] must be np.ndarray, got type={type(primary_frame)}."
            )
        return primary_frame

    @staticmethod
    def _extract_wrist_frames(ex, *, ex_idx: int, required: bool) -> List[np.ndarray]:
        if "wrist_image" not in ex:
            if required:
                raise ValueError(
                    f"examples[{ex_idx}] missing required key 'wrist_image' "
                    "when `framework.action_model.enable_wrist_view=true`."
                )
            return []

        wrist_image = ex["wrist_image"]
        if not isinstance(wrist_image, (list, tuple)):
            raise ValueError(
                f"examples[{ex_idx}]['wrist_image'] must be a list/tuple, got type={type(wrist_image)}."
            )
        if len(wrist_image) < 1:
            raise ValueError(
                f"examples[{ex_idx}]['wrist_image'] must contain at least 1 wrist view, got {len(wrist_image)}."
            )

        wrist_frames: List[np.ndarray] = []
        for wrist_idx, wrist_frame in enumerate(wrist_image):
            if not isinstance(wrist_frame, np.ndarray):
                raise ValueError(
                    f"examples[{ex_idx}]['wrist_image'][{wrist_idx}] must be np.ndarray, "
                    f"got type={type(wrist_frame)}."
                )
            wrist_frames.append(wrist_frame)
        return wrist_frames

    def _build_qwen_inputs(
        self,
        *,
        image_views_batch: Sequence[Sequence[np.ndarray]],
        wrist_image_views_batch: Sequence[Sequence[np.ndarray]],
        instructions: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        processed_image_views_batch: List[List[Image.Image]] = []
        for image_views in image_views_batch:
            processed_image_views_batch.append(
                [_preprocess_numpy_frame_to_pil(frame, target_hw=self.lam_image_hw) for frame in image_views]
            )

        processed_wrist_image_views_batch: List[List[Image.Image]] = []
        for image_views in wrist_image_views_batch:
            processed_wrist_image_views_batch.append(
                [_preprocess_numpy_frame_to_pil(frame, target_hw=self.lam_image_hw) for frame in image_views]
            )

        return self.policy_vlm_adapter.build_qwenvl_inputs(
            images=processed_image_views_batch,
            wrist_images=processed_wrist_image_views_batch,
            instructions=instructions,
        )


class LatentWorldPolicyTrainBatchBuilder(_LatentWorldPolicyBatchBuilderBase):
    def build_train_batch(self, examples: Sequence[LatentWorldPolicyTrainExample]) -> LatentWorldPolicyTrainBatch:
        enable_wrist_view = bool(self.policy_cfg.enable_wrist_view)

        image_views_batch: List[List[np.ndarray]] = []
        wrist_image_views_batch: List[List[np.ndarray]] = []
        primary_video_seqs: List[List[np.ndarray]] = []
        instructions: List[str] = []
        actions_list: List[torch.Tensor] = []
        states_list: List[torch.Tensor] = []
        embodiment_ids: List[int] = []
        action_hz_list: List[float] = []

        for ex_idx, ex in enumerate(examples):
            for key in ("primary_image", "primary_video", "lang", "state", "action", "embodiment_id", "action_hz"):
                if key not in ex:
                    raise KeyError(f"training examples[{ex_idx}] missing required key '{key}'.")

            instruction = ex["lang"]
            primary_frame = self._extract_primary_frame(ex["primary_image"], ex_idx=ex_idx)

            action_hz = float(ex["action_hz"])
            if action_hz <= 0.0:
                raise ValueError(f"training examples[{ex_idx}]['action_hz'] must be > 0, got {action_hz}.")

            embodiment_id = int(ex["embodiment_id"])
            state_tensor = self._to_2d_sequence_tensor(ex["state"], field_name="state", ex_idx=ex_idx)
            action_tensor = self._to_2d_sequence_tensor(ex["action"], field_name="action", ex_idx=ex_idx)

            primary_video_views = ex["primary_video"]
            if not isinstance(primary_video_views, (list, tuple)):
                raise ValueError(
                    f"training examples[{ex_idx}]['primary_video'] must be a list/tuple, "
                    f"got type={type(primary_video_views)}."
                )
            if len(primary_video_views) != 1:
                raise ValueError(
                    f"training examples[{ex_idx}]['primary_video'] must contain exactly 1 primary view, "
                    f"got {len(primary_video_views)}."
                )
            primary_video_seq = primary_video_views[0]
            if not isinstance(primary_video_seq, (list, tuple)) or len(primary_video_seq) == 0:
                raise ValueError(
                    f"training examples[{ex_idx}] primary video sequence must be non-empty, "
                    f"got type={type(primary_video_seq)}."
                )

            wrist_frames = self._extract_wrist_frames(ex, ex_idx=ex_idx, required=enable_wrist_view)

            image_views_batch.append([primary_frame])
            wrist_image_views_batch.append(wrist_frames)
            primary_video_seqs.append(list(primary_video_seq))
            instructions.append(instruction)
            actions_list.append(action_tensor)
            states_list.append(state_tensor)
            embodiment_ids.append(embodiment_id)
            action_hz_list.append(action_hz)

        qwen_inputs = self._build_qwen_inputs(
            image_views_batch=image_views_batch,
            wrist_image_views_batch=wrist_image_views_batch,
            instructions=instructions,
        )
        input_ids = qwen_inputs["input_ids"]
        act_mask, flow_mask = self._build_masks(
            input_ids,
            act_queries=int(self.policy_backend.num_action_queries),
            flow_queries=int(self.policy_backend.flow.flow_action_query.shape[0]),
            placeholder_id=int(self.policy_backend.placeholder_token_id),
        )

        action_dim = int(self.policy_cfg.flow_cfg.action_dim)
        window_size = int(self.policy_cfg.action_horizon)
        state_dim = int(self.policy_cfg.flow_cfg.state_dim)

        action_tensors = []
        action_masks = []
        state_tensors = []
        state_masks = []

        for idx, state_tensor in enumerate(states_list):
            action_tensor = actions_list[idx]
            action_tensor, action_dim_mask = self._align_feature_dim_with_mask(action_tensor, action_dim)
            action_tensor, action_time_mask, was_downsampled = self._sample_or_pad_sequence_with_mask(action_tensor, window_size)
            if was_downsampled:
                action_hz_list[idx] *= 0.5
            action_tensors.append(action_tensor)
            action_masks.append(action_time_mask.unsqueeze(1) & action_dim_mask.unsqueeze(0))

            aligned_state, state_dim_mask = self._align_feature_dim_with_mask(state_tensor, state_dim)
            state_tensors.append(aligned_state[-1])
            state_masks.append(state_dim_mask)

        primary_video_tensors = [
            _numpy_video_to_tchw_float(primary_video_seq, target_hw=self.lam_image_hw)
            for primary_video_seq in primary_video_seqs
        ]

        batch: LatentWorldPolicyTrainBatch = {
            "pixel_values": qwen_inputs["pixel_values"],
            "input_ids": input_ids,
            "attention_mask": qwen_inputs["attention_mask"],
            "act_placeholder_mask": act_mask,
            "flow_placeholder_mask": flow_mask,
            "primary_video": torch.stack(primary_video_tensors, dim=0),
            "state": torch.stack(state_tensors, dim=0),
            "state_mask": torch.stack(state_masks, dim=0),
            "embodiment_id": torch.tensor(embodiment_ids, dtype=torch.long),
            "action_hz": torch.tensor(action_hz_list, dtype=torch.float32),
            "image_grid_thw": qwen_inputs.get("image_grid_thw"),
            "actions": torch.stack(action_tensors, dim=0),
            "actions_mask": torch.stack(action_masks, dim=0),
        }
        if not torch.is_tensor(batch["image_grid_thw"]):
            batch["image_grid_thw"] = None
        return batch


class LatentWorldPolicyInferBatchBuilder(_LatentWorldPolicyBatchBuilderBase):
    _ALLOWED_INFER_KEYS = {
        "lang",
        "primary_image",
        "action_hz",
        "embodiment_id",
        "state",
        "wrist_image",
    }
    _REQUIRED_INFER_KEYS = {"lang", "primary_image", "action_hz", "embodiment_id"}

    def build_infer_batch(self, examples: Sequence[LatentWorldPolicyInferExample]) -> LatentWorldPolicyInferBatch:
        enable_wrist_view = bool(self.policy_cfg.enable_wrist_view)
        state_dim = int(self.policy_cfg.flow_cfg.state_dim)

        image_views_batch: List[List[np.ndarray]] = []
        wrist_image_views_batch: List[List[np.ndarray]] = []
        primary_image_tensors: List[torch.Tensor] = []
        instructions: List[str] = []
        state_tensors: List[torch.Tensor] = []
        state_masks: List[torch.Tensor] = []
        embodiment_ids: List[int] = []
        action_hz_list: List[float] = []

        for ex_idx, ex in enumerate(examples):
            extra_keys = sorted(set(ex.keys()) - self._ALLOWED_INFER_KEYS)
            if extra_keys:
                raise KeyError(
                    "inference examples[{idx}] contains unsupported keys {keys}. "
                    "Allowed keys are: {allowed}.".format(
                        idx=ex_idx,
                        keys=extra_keys,
                        allowed=sorted(self._ALLOWED_INFER_KEYS),
                    )
                )
            missing_keys = sorted(self._REQUIRED_INFER_KEYS - set(ex.keys()))
            if missing_keys:
                raise KeyError(
                    f"inference examples[{ex_idx}] missing required keys {missing_keys}. "
                    f"Required keys are: {sorted(self._REQUIRED_INFER_KEYS)}."
                )

            instruction = ex["lang"]
            primary_frame = self._extract_primary_frame(ex["primary_image"], ex_idx=ex_idx)
            wrist_frames = self._extract_wrist_frames(ex, ex_idx=ex_idx, required=enable_wrist_view)

            action_hz = float(ex["action_hz"])
            if action_hz <= 0.0:
                raise ValueError(f"inference examples[{ex_idx}]['action_hz'] must be > 0, got {action_hz}.")
            embodiment_id = int(ex["embodiment_id"])

            if "state" in ex:
                state_seq = self._to_2d_sequence_tensor(ex["state"], field_name="state", ex_idx=ex_idx)
                aligned_state, state_dim_mask = self._align_feature_dim_with_mask(state_seq, state_dim)
                state_tensors.append(aligned_state[-1])
                state_masks.append(state_dim_mask)
            else:
                state_tensors.append(torch.zeros((state_dim,), dtype=torch.float32))
                state_masks.append(torch.zeros((state_dim,), dtype=torch.bool))

            image_views_batch.append([primary_frame])
            wrist_image_views_batch.append(wrist_frames)
            primary_image_tensors.append(_numpy_image_to_chw_float(primary_frame, target_hw=self.lam_image_hw))
            instructions.append(instruction)
            embodiment_ids.append(embodiment_id)
            action_hz_list.append(action_hz)

        qwen_inputs = self._build_qwen_inputs(
            image_views_batch=image_views_batch,
            wrist_image_views_batch=wrist_image_views_batch,
            instructions=instructions,
        )

        input_ids = qwen_inputs["input_ids"]
        act_mask, flow_mask = self._build_masks(
            input_ids,
            act_queries=int(self.policy_backend.num_action_queries),
            flow_queries=int(self.policy_backend.flow.flow_action_query.shape[0]),
            placeholder_id=int(self.policy_backend.placeholder_token_id),
        )

        batch: LatentWorldPolicyInferBatch = {
            "pixel_values": qwen_inputs["pixel_values"],
            "input_ids": input_ids,
            "attention_mask": qwen_inputs["attention_mask"],
            "act_placeholder_mask": act_mask,
            "flow_placeholder_mask": flow_mask,
            "primary_image": torch.stack(primary_image_tensors, dim=0),
            "state": torch.stack(state_tensors, dim=0),
            "state_mask": torch.stack(state_masks, dim=0),
            "embodiment_id": torch.tensor(embodiment_ids, dtype=torch.long),
            "action_hz": torch.tensor(action_hz_list, dtype=torch.float32),
            "image_grid_thw": qwen_inputs.get("image_grid_thw"),
        }
        if not torch.is_tensor(batch["image_grid_thw"]):
            batch["image_grid_thw"] = None
        return batch
