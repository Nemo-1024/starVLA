from __future__ import annotations

import re
from typing import List, Sequence, Tuple, Union, cast

import numpy as np
import torch
from PIL import Image

from .types import LiberoExample, LatentWorldPolicyInferBatch, LatentWorldPolicyTrainBatch

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
    video_tchw = video_thwc.permute(0, 3, 1, 2).to(dtype=torch.float32).div_(255.0)
    return video_tchw


class LatentWorldPolicyBatchBuilder:
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
    def _sample_or_pad_sequence(seq: torch.Tensor, target_len: int) -> torch.Tensor:
        if seq.shape[0] == target_len:
            return seq
        if seq.shape[0] > target_len:
            idx = torch.linspace(0, seq.shape[0] - 1, target_len).round().long()
            return seq.index_select(0, idx)
        pad_len = target_len - seq.shape[0]
        pad = seq[-1:].repeat(pad_len, 1)
        return torch.cat([seq, pad], dim=0)

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
    def _sample_or_pad_sequence_with_mask(seq: torch.Tensor, target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq.shape[0] == target_len:
            return seq, torch.ones((target_len,), dtype=torch.bool)
        if seq.shape[0] > target_len:
            idx = torch.linspace(0, seq.shape[0] - 1, target_len).round().long()
            return seq.index_select(0, idx), torch.ones((target_len,), dtype=torch.bool)
        pad_len = target_len - seq.shape[0]
        pad = seq[-1:].repeat(pad_len, 1)
        out = torch.cat([seq, pad], dim=0)
        mask = torch.zeros((target_len,), dtype=torch.bool)
        mask[: seq.shape[0]] = True
        return out, mask

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

    def build_train_batch(self, examples: Sequence[LiberoExample]) -> LatentWorldPolicyTrainBatch:
        return cast(LatentWorldPolicyTrainBatch, self._build_batch(examples, include_actions=True))

    def build_infer_batch(self, examples: Sequence[LiberoExample]) -> LatentWorldPolicyInferBatch:
        return cast(LatentWorldPolicyInferBatch, self._build_batch(examples, include_actions=False))

    def _build_batch(
        self,
        examples: Sequence[LiberoExample],
        *,
        include_actions: bool,
    ) -> Union[LatentWorldPolicyInferBatch, LatentWorldPolicyTrainBatch]:
        enable_wrist_view = bool(self.policy_cfg.enable_wrist_view)

        image_views_batch: List[List[np.ndarray]] = []
        primary_video_seqs: List[List[np.ndarray]] = []
        wrist_video_seqs: List[List[np.ndarray]] = []
        instructions: List[str] = []
        actions_list: List[torch.Tensor] = []
        states_list: List[torch.Tensor] = []
        embodiment_ids: List[int] = []

        for ex_idx, ex in enumerate(examples):
            image_views = ex["image"]
            instruction = ex["lang"]
            state_tensor = ex["state"]
            embodiment_id = ex["embodiment_id"]

            # if not isinstance(instruction, str):
            #     raise TypeError(f"examples[{ex_idx}]['lang'] must be str, got type={type(instruction)}.")
            # instruction = self._formalize_language(instruction)

            if not isinstance(image_views, (list, tuple)) or len(image_views) == 0:
                raise ValueError(
                    f"examples[{ex_idx}]['image'] must be a non-empty list/tuple, got type={type(image_views)}."
                )

            action_tensor = None
            if include_actions:
                if "action" not in ex:
                    raise KeyError(f"examples[{ex_idx}] missing required key 'action' for training.")
                action_tensor = self._to_2d_sequence_tensor(ex["action"], field_name="action", ex_idx=ex_idx)

            video_views = None
            if "primary_videos" in ex:
                video_views = ex["primary_videos"]
            elif "video" in ex:
                video_views = ex["video"]

            if video_views is None:
                if include_actions:
                    raise KeyError(
                        f"examples[{ex_idx}] missing required key 'primary_videos'/'video' for training."
                    )
                primary_video_seq = [image_views[0]]
            else:
                if not isinstance(video_views, (list, tuple)) or len(video_views) == 0:
                    raise ValueError(
                        f"examples[{ex_idx}] video views must be a non-empty list/tuple, got type={type(video_views)}."
                    )
                primary_video_seq = video_views[0]
                if not isinstance(primary_video_seq, (list, tuple)) or len(primary_video_seq) == 0:
                    raise ValueError(
                        f"examples[{ex_idx}] primary video sequence must be non-empty, got type={type(primary_video_seq)}."
                    )

            if enable_wrist_view:
                if "wrist_images" not in ex:
                    raise ValueError(
                        f"examples[{ex_idx}] missing required key 'wrist_images' "
                        "when `framework.action_model.enable_wrist_view=true`."
                    )
                wrist_images = ex["wrist_images"]
                if not isinstance(wrist_images, (list, tuple)):
                    raise ValueError(
                        f"examples[{ex_idx}]['wrist_images'] must be a list/tuple, got type={type(wrist_images)}."
                    )
                if len(wrist_images) != 1:
                    raise ValueError(
                        f"examples[{ex_idx}]['wrist_images'] must contain exactly 1 wrist view, got {len(wrist_images)}."
                    )
                wrist_frame = wrist_images[0]
                if not isinstance(wrist_frame, np.ndarray):
                    raise ValueError(
                        f"examples[{ex_idx}]['wrist_images'][0] must be np.ndarray, got type={type(wrist_frame)}."
                    )
                wrist_video_seqs.append([wrist_frame])

            image_views_batch.append(image_views)
            primary_video_seqs.append(list(primary_video_seq))
            instructions.append(instruction)
            if include_actions:
                assert action_tensor is not None
                actions_list.append(action_tensor)
            states_list.append(self._to_2d_sequence_tensor(state_tensor, field_name="state", ex_idx=ex_idx))
            embodiment_ids.append(embodiment_id)

        processed_image_views_batch: List[List[Image.Image]] = []
        for image_views in image_views_batch:
            processed_image_views = [
                _preprocess_numpy_frame_to_pil(frame, target_hw=self.lam_image_hw)
                for frame in image_views
            ]
            processed_image_views_batch.append(processed_image_views)

        qwen_inputs = self.policy_vlm_adapter.build_qwenvl_inputs(
            images=processed_image_views_batch,
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
        lam_num_frames = int(self.policy_backend.lam.num_frames)

        action_tensors = []
        action_masks = []
        state_tensors = []
        state_masks = []

        for idx, state_tensor in enumerate(states_list):
            if include_actions:
                action_tensor = actions_list[idx]
                action_tensor, action_dim_mask = self._align_feature_dim_with_mask(action_tensor, action_dim)
                action_tensor, action_time_mask = self._sample_or_pad_sequence_with_mask(action_tensor, window_size)
                action_mask = action_time_mask.unsqueeze(1) & action_dim_mask.unsqueeze(0)
                action_tensors.append(action_tensor)
                action_masks.append(action_mask)

            state_tensor, state_dim_mask = self._align_feature_dim_with_mask(state_tensor, state_dim)
            state_seq = self._sample_or_pad_sequence(state_tensor, lam_num_frames)
            state_tensors.append(state_seq[-1])
            state_masks.append(state_dim_mask)

        primary_tensors = []
        for primary_video_seq in primary_video_seqs:
            primary_video_tchw = _numpy_video_to_tchw_float(primary_video_seq, target_hw=self.lam_image_hw)
            primary_tensors.append(primary_video_tchw)

        lam_videos = torch.stack(primary_tensors, dim=0)
        state = torch.stack(state_tensors, dim=0)
        state_mask = torch.stack(state_masks, dim=0)

        wrist_videos = None
        if enable_wrist_view:
            wrist_tensors = []
            for wrist_video_seq in wrist_video_seqs:
                wrist_video_tchw = _numpy_video_to_tchw_float(wrist_video_seq, target_hw=self.lam_image_hw)
                wrist_tensors.append(wrist_video_tchw)
            wrist_videos = torch.stack(wrist_tensors, dim=0)

        infer_batch: LatentWorldPolicyInferBatch = {
            "pixel_values": qwen_inputs["pixel_values"],
            "input_ids": input_ids,
            "attention_mask": qwen_inputs["attention_mask"],
            "act_placeholder_mask": act_mask,
            "flow_placeholder_mask": flow_mask,
            "lam_videos": lam_videos,
            "state": state,
            "state_mask": state_mask,
            "embodiment_id": torch.tensor(embodiment_ids, dtype=torch.long),
            "image_grid_thw": qwen_inputs.get("image_grid_thw"),
            "wrist_videos": wrist_videos,
        }

        if not torch.is_tensor(infer_batch["image_grid_thw"]):
            infer_batch["image_grid_thw"] = None

        if include_actions:
            train_batch = cast(LatentWorldPolicyTrainBatch, dict(infer_batch))
            train_batch["actions"] = torch.stack(action_tensors, dim=0)
            train_batch["actions_mask"] = torch.stack(action_masks, dim=0)
            return train_batch
        return infer_batch
