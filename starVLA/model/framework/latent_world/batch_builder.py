from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


def _pil_video_to_tchw_float(video_frames: List[Image.Image]) -> torch.Tensor:
    if len(video_frames) == 0:
        raise ValueError("Video frame list cannot be empty.")
    frame_arrays = []
    for frame_idx, frame in enumerate(video_frames):
        if not isinstance(frame, Image.Image):
            raise TypeError(f"`video` frame {frame_idx} must be PIL.Image.Image, got {type(frame)}.")
        frame_arrays.append(torch.from_numpy(np.asarray(frame.convert("RGB"))))
    video_thwc = torch.stack(frame_arrays, dim=0)  # [T, H, W, C], uint8
    video_tchw = video_thwc.permute(0, 3, 1, 2).to(dtype=torch.float32).div_(255.0)
    return video_tchw


class LatentWorldPromptBuilder:
    def __init__(self, *, placeholder_token: str, act_queries: int, flow_queries: int) -> None:
        self.placeholder_token = placeholder_token
        self.act_queries = int(act_queries)
        self.flow_queries = int(flow_queries)
        self.placeholder_block = " ".join([self.placeholder_token] * (self.act_queries + self.flow_queries))

    def build(self, instructions: List[str]) -> List[str]:
        return [f"{self.placeholder_block}\n{instruction}" for instruction in instructions]


class LatentWorldBatchBuilder:
    def __init__(
        self,
        *,
        model_cfg,
        world_model,
        qwen_vl_interface,
        prompt_builder: LatentWorldPromptBuilder,
        lam_image_hw: Tuple[int, int],
    ) -> None:
        self.model_cfg = model_cfg
        self.world_model = world_model
        self.qwen_vl_interface = qwen_vl_interface
        self.prompt_builder = prompt_builder
        self.lam_image_hw = (int(lam_image_hw[0]), int(lam_image_hw[1]))

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
    def _build_masks(
        input_ids: torch.Tensor,
        *,
        act_queries: int,
        flow_queries: int,
        placeholder_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, _ = input_ids.shape
        expected = int(act_queries + flow_queries)
        placeholder = input_ids == int(placeholder_id)
        counts = placeholder.sum(dim=1)
        bad = torch.nonzero(counts < expected, as_tuple=False).flatten()
        if bad.numel() > 0:
            b = int(bad[0].item())
            raise ValueError(
                f"Insufficient placeholder tokens in sample {b}: got={int(counts[b].item())}, expected={expected}"
            )
        order = placeholder.cumsum(dim=1)
        act_mask = placeholder & (order <= int(act_queries))
        flow_mask = placeholder & (order > int(act_queries)) & (order <= expected)
        if not torch.all(act_mask.sum(dim=1) == int(act_queries)):
            raise ValueError("Failed to build action placeholder mask with expected query count.")
        if not torch.all(flow_mask.sum(dim=1) == int(flow_queries)):
            raise ValueError("Failed to build flow placeholder mask with expected query count.")
        return act_mask, flow_mask

    @staticmethod
    def _ensure_image_views(image_views: object, sample_idx: int) -> List[Image.Image]:
        if not isinstance(image_views, list):
            raise TypeError(f"Sample {sample_idx} `image` must be List[PIL.Image.Image].")
        if len(image_views) == 0:
            raise ValueError(f"Sample {sample_idx} `image` cannot be empty.")
        for idx, image in enumerate(image_views):
            if not isinstance(image, Image.Image):
                raise TypeError(
                    f"Sample {sample_idx} `image[{idx}]` must be PIL.Image.Image, got {type(image)}."
                )
        return image_views

    @staticmethod
    def _ensure_video_views(video_views: object, sample_idx: int, name: str) -> List[List[Image.Image]]:
        if not isinstance(video_views, list):
            raise TypeError(f"Sample {sample_idx} `{name}` must be List[List[PIL.Image.Image]].")
        if len(video_views) == 0:
            raise ValueError(f"Sample {sample_idx} `{name}` cannot be empty.")
        for view_idx, view in enumerate(video_views):
            if not isinstance(view, list):
                raise TypeError(
                    f"Sample {sample_idx} `{name}[{view_idx}]` must be List[PIL.Image.Image], got {type(view)}."
                )
            if len(view) == 0:
                raise ValueError(f"Sample {sample_idx} `{name}[{view_idx}]` cannot be empty.")
            for frame_idx, frame in enumerate(view):
                if not isinstance(frame, Image.Image):
                    raise TypeError(
                        f"Sample {sample_idx} `{name}[{view_idx}][{frame_idx}]` must be PIL.Image.Image, "
                        f"got {type(frame)}."
                    )
        return video_views

    @staticmethod
    def _ensure_2d_tensor(tensor: object, name: str, sample_idx: int) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Sample {sample_idx} `{name}` must be torch.Tensor.")
        if tensor.ndim != 2:
            raise ValueError(f"Sample {sample_idx} `{name}` must be 2D tensor, got shape {tuple(tensor.shape)}.")
        return tensor

    @staticmethod
    def _ensure_embodiment_id(value: object, sample_idx: int) -> int:
        if value is None:
            raise ValueError(f"Sample {sample_idx} missing required `embodiment_id`.")
        if isinstance(value, bool) or type(value) is not int:
            raise TypeError(
                f"Sample {sample_idx} `embodiment_id` must be Python int, got {type(value)}."
            )
        return value

    def build(self, examples: Sequence[dict], *, require_actions: bool) -> Dict[str, torch.Tensor]:
        if not isinstance(examples, list):
            raise TypeError("`examples` must be List[dict].")
        if len(examples) == 0:
            raise ValueError("Received empty batch for LatentWorldVLA training.")
        if bool(self.model_cfg.enable_wrist_view):
            raise ValueError(
                "Current data contract does not provide wrist videos. "
                "Set `framework.latent_world.enable_wrist_view=false`."
            )

        image_views_batch: List[List[Image.Image]] = []
        primary_video_seqs: List[List[Image.Image]] = []
        instructions: List[str] = []
        actions_list: List[torch.Tensor] = []
        states_list: List[torch.Tensor] = []
        embodiment_ids: List[int] = []

        for sample_idx, ex in enumerate(examples):
            if not isinstance(ex, dict):
                raise TypeError(f"Sample {sample_idx} must be dict.")

            image_views = self._ensure_image_views(ex["image"], sample_idx)
            video_key = "primary_videos" if "primary_videos" in ex else "video"
            video_views = self._ensure_video_views(ex[video_key], sample_idx, video_key)
            if not isinstance(ex["lang"], str):
                raise TypeError(f"Sample {sample_idx} `lang` must be str.")
            action_tensor = self._ensure_2d_tensor(ex["action"], "action", sample_idx)
            state_tensor = self._ensure_2d_tensor(ex["state"], "state", sample_idx)
            embodiment_id = self._ensure_embodiment_id(ex.get("embodiment_id"), sample_idx)

            image_views_batch.append(image_views)
            primary_video_seqs.append(video_views[0])
            instructions.append(ex["lang"])
            actions_list.append(action_tensor.detach().to(device="cpu", dtype=torch.float32))
            states_list.append(state_tensor.detach().to(device="cpu", dtype=torch.float32))
            embodiment_ids.append(embodiment_id)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=image_views_batch,
            instructions=self.prompt_builder.build(instructions),
        )

        input_ids = qwen_inputs["input_ids"]
        act_mask, flow_mask = self._build_masks(
            input_ids,
            act_queries=int(self.world_model.num_action_queries),
            flow_queries=int(self.world_model.flow.flow_action_query.shape[0]),
            placeholder_id=int(self.world_model.placeholder_token_id),
        )

        action_dim = int(self.model_cfg.flow_cfg.action_dim)
        window_size = int(self.model_cfg.flow_cfg.window_size)
        state_dim = int(self.model_cfg.flow_cfg.state_dim)
        lam_num_frames = int(self.world_model.lam.num_frames)

        action_tensors = []
        state_tensors = []

        for action_tensor, state_tensor in zip(actions_list, states_list):
            if require_actions:
                if action_tensor.shape[1] < action_dim:
                    pad = torch.zeros(
                        action_tensor.shape[0],
                        action_dim - action_tensor.shape[1],
                        dtype=torch.float32,
                    )
                    action_tensor = torch.cat([action_tensor, pad], dim=1)
                elif action_tensor.shape[1] > action_dim:
                    action_tensor = action_tensor[:, :action_dim]
                if action_tensor.shape[0] < window_size:
                    action_tensor = self._sample_or_pad_sequence(action_tensor, window_size)
                elif action_tensor.shape[0] > window_size:
                    action_tensor = action_tensor[-window_size:]
                action_tensors.append(action_tensor)

            if state_tensor.shape[1] < state_dim:
                pad = torch.zeros(
                    state_tensor.shape[0],
                    state_dim - state_tensor.shape[1],
                    dtype=torch.float32,
                )
                state_tensor = torch.cat([state_tensor, pad], dim=1)
            elif state_tensor.shape[1] > state_dim:
                state_tensor = state_tensor[:, :state_dim]
            state_seq = self._sample_or_pad_sequence(state_tensor, lam_num_frames)
            state_tensors.append(state_seq[-1])

        primary_tensors = []
        for sample_idx, primary_video_seq in enumerate(primary_video_seqs):
            if len(primary_video_seq) != lam_num_frames:
                raise ValueError(
                    f"Sample {sample_idx} primary video length mismatch: got={len(primary_video_seq)}, "
                    f"expected={lam_num_frames}. Please align `datasets.vla_data.num_frames` with LAM config."
                )
            primary_video_tchw = _pil_video_to_tchw_float(primary_video_seq)
            expected_hw = self.lam_image_hw
            actual_hw = tuple(primary_video_tchw.shape[-2:])
            if actual_hw != expected_hw:
                raise ValueError(
                    f"Sample {sample_idx} primary video resolution mismatch: got={actual_hw}, "
                    f"expected={expected_hw}. Please align dataset preprocessing to the fixed (H,W)=(256,256)."
                )
            primary_tensors.append(primary_video_tchw)

        lam_videos = torch.stack(primary_tensors, dim=0)
        state = torch.stack(state_tensors, dim=0)

        wrist_videos = None

        device = next(self.world_model.parameters()).device
        batch = {
            "pixel_values": qwen_inputs["pixel_values"].to(device=device, non_blocking=True),
            "input_ids": input_ids.to(device=device, non_blocking=True),
            "attention_mask": qwen_inputs["attention_mask"].to(device=device, non_blocking=True),
            "act_placeholder_mask": act_mask.to(device=device, non_blocking=True),
            "flow_placeholder_mask": flow_mask.to(device=device, non_blocking=True),
            "lam_videos": lam_videos.to(device=device, non_blocking=True),
            "state": state.to(device=device, non_blocking=True),
            "embodiment_id": torch.tensor(embodiment_ids, dtype=torch.long).to(
                device=device, non_blocking=True
            ),
            "image_grid_thw": qwen_inputs.get("image_grid_thw"),
            "wrist_videos": wrist_videos.to(device=device) if wrist_videos is not None else None,
        }

        if torch.is_tensor(batch["image_grid_thw"]):
            batch["image_grid_thw"] = batch["image_grid_thw"].to(device=device, non_blocking=True)
        else:
            batch["image_grid_thw"] = None

        if require_actions:
            batch["actions"] = torch.stack(action_tensors, dim=0).to(device=device, non_blocking=True)

        return batch
