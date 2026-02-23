import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.data_config_lam import (
    build_lam_state_normalize_transform,
    filter_lam_video_keys,
)
from starVLA.dataloader.gr00t_lerobot.image_preprocess import (
    preprocess_video_frames_nhwc,
)
from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
    ModalityConfig,
)
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG


def _build_modality_config(
    robot_type: str,
    num_frames: int,
    preferred_video_key: Optional[str] = None,
    state_keys: Optional[Sequence[str]] = None,
    include_action: bool = False,
    include_language: bool = False,
) -> Dict[str, ModalityConfig]:
    """
    Clone ROBOT_TYPE_CONFIG_MAP[robot_type].modality_config() and override delta_indices.
    We only keep video/state by default to reduce IO for LAM.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    base_cfg = ROBOT_TYPE_CONFIG_MAP[robot_type].modality_config()
    default_delta = list(range(num_frames))

    def override_delta(cfg: ModalityConfig, keys: Optional[Sequence[str]]) -> ModalityConfig:
        new_keys = list(keys) if keys is not None else cfg.modality_keys
        return ModalityConfig(delta_indices=list(default_delta), modality_keys=new_keys)

    # video
    video_keys = filter_lam_video_keys(
        base_video_keys=base_cfg["video"].modality_keys,
        preferred_video_key=preferred_video_key,
    )
    video_modality = override_delta(base_cfg["video"], video_keys)

    # state
    state_modality = override_delta(base_cfg["state"], state_keys)

    modality_configs: Dict[str, ModalityConfig] = {
        "video": video_modality,
        "state": state_modality,
    }

    if include_action and "action" in base_cfg:
        modality_configs["action"] = override_delta(base_cfg["action"], base_cfg["action"].modality_keys)
    if include_language and "language" in base_cfg:
        modality_configs["language"] = base_cfg["language"]

    return modality_configs


def _build_temporal_delta_indices(num_frames: int, frame_dt_sec: float, fps: float) -> np.ndarray:
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if frame_dt_sec <= 0:
        raise ValueError(f"frame_dt_sec must be > 0, got {frame_dt_sec}")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    stride = max(1, int(round(frame_dt_sec * fps)))
    return np.arange(0, num_frames * stride, stride, dtype=np.int64)


class LeRobotLAMDataset(Dataset):
    """
    Mixture dataset that reuses starVLA's LeRobot loaders but emits LAM-ready raw samples.

    Output sample:
        {
            "frames": torch.Tensor[T, H, W, C] uint8,
            "proprio": torch.Tensor[T, D] float32,
            "embodiment_id": int,
        }
    """

    def __init__(
        self,
        data_root_dir: str | Path,
        data_mix: str,
        num_frames: int,
        video_backend: str = "pyav",
        preferred_video_key: Optional[str] = None,
        state_keys: Optional[Sequence[str]] = None,
        *,
        frame_dt_sec: float,
        max_retries: int = 5,
        debug_repeat_batch: Union[bool, int] = False,
    ) -> None:
        super().__init__()
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        if frame_dt_sec <= 0:
            raise ValueError(f"frame_dt_sec must be > 0, got {frame_dt_sec}")

        self.data_root_dir = Path(data_root_dir)
        self.data_mix = data_mix
        self.num_frames = num_frames
        self.preferred_video_key = preferred_video_key
        self.state_keys = list(state_keys) if state_keys else None
        self.frame_dt_sec = frame_dt_sec
        self.max_retries = max_retries

        # "auto" 选择在不同环境里更稳健：
        # - 优先 decord（通常最快）
        # - 其次 torchcodec
        # - 否则回退到 torchvision_av

        self.video_backend = video_backend
        
        # Debug mode: cache and repeat samples
        self.debug_repeat_batch = debug_repeat_batch
        self._cached_samples: List[Dict] = []
        self._cache_initialized: bool = False

        data_cfg = {
            "data_root_dir": str(self.data_root_dir),
            "video_backend": self.video_backend,
        }

        mixture_spec = DATASET_NAMED_MIXTURES[self.data_mix]
        # dedupe (dataset_name, robot_type)
        seen: set[Tuple[str, str]] = set()
        dataset_mixture: List[Tuple[LeRobotSingleDataset, float]] = []

        for dataset_name, weight, robot_type in mixture_spec:
            key = (dataset_name, robot_type)
            if key in seen:
                continue
            seen.add(key)

            modality_cfg = _build_modality_config(
                robot_type=robot_type,
                num_frames=self.num_frames,
                preferred_video_key=self.preferred_video_key,
                state_keys=self.state_keys,
                include_action=False,
                include_language=False,
            )
            # Map robot_type to EmbodimentTag
            embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG.get(robot_type)
            if embodiment_tag is None:
                raise ValueError(f"Robot type '{robot_type}' not found in ROBOT_TYPE_TO_EMBODIMENT_TAG mapping")

            lam_transform = build_lam_state_normalize_transform(
                robot_type=robot_type,
                state_keys=modality_cfg["state"].modality_keys,
            )

            ds = LeRobotSingleDataset(
                dataset_path=self.data_root_dir / dataset_name,
                modality_configs=modality_cfg,
                embodiment_tag=embodiment_tag,
                video_backend=self.video_backend,
                transforms=lam_transform,
                data_cfg=data_cfg,
            )
            self._apply_temporal_delta_indices(ds)
            dataset_mixture.append((ds, weight))

        self.mixture = LeRobotMixtureDataset(
            dataset_mixture,
            mode="train",
            balance_dataset_weights=True,
            seed=42,
            data_cfg=data_cfg,
        )

    def _apply_temporal_delta_indices(self, dataset: LeRobotSingleDataset) -> None:
        video_keys = dataset.modality_keys.get("video", [])
        if not video_keys:
            raise RuntimeError(f"Dataset {dataset.dataset_name} has no video keys configured.")

        # Use preferred view if present; otherwise fallback to the first configured view.
        video_key_full = (
            self.preferred_video_key
            if self.preferred_video_key and self.preferred_video_key in video_keys
            else video_keys[0]
        )
        video_subkey = video_key_full.replace("video.", "")
        fps = float(dataset.metadata.modalities.video[video_subkey].fps)
        delta_arr = _build_temporal_delta_indices(
            num_frames=self.num_frames,
            frame_dt_sec=self.frame_dt_sec,
            fps=fps,
        )

        for k in video_keys:
            dataset._delta_indices[k] = delta_arr.copy()
        for k in dataset.modality_keys.get("state", []):
            dataset._delta_indices[k] = delta_arr.copy()

    def __len__(self) -> int:
        # In debug mode, return a large virtual length to support multiple epochs
        if self.debug_repeat_batch:
            return 10000
        return len(self.mixture)

    def _try_get_sample(self, index: int) -> Dict:
        dataset, traj_id, base_index = self.mixture.sample_step(index)
        raw_data = dataset.get_step_data(traj_id, base_index)
        data = dataset.transforms(raw_data)

        # video
        available_video_keys = dataset.modality_keys["video"]
        if self.preferred_video_key and self.preferred_video_key in available_video_keys:
            video_key = self.preferred_video_key
        else:
            video_key = available_video_keys[0]
        frames = data[video_key]  # (T, H, W, C)
        frames = preprocess_video_frames_nhwc(frames)
        if isinstance(frames, torch.Tensor):
            frames_t = frames
            if frames_t.dtype != torch.uint8:
                frames_t = frames_t.to(torch.uint8)
        else:
            frames_t = torch.from_numpy(np.ascontiguousarray(frames)).to(torch.uint8)

        # state (proprio)
        proprio_tensors: List[torch.Tensor] = []
        for state_key in dataset.modality_keys.get("state", []):
            state_value = data[state_key]
            if isinstance(state_value, torch.Tensor):
                state_tensor = state_value
            else:
                state_tensor = torch.from_numpy(np.asarray(state_value))
            proprio_tensors.append(state_tensor.to(torch.float32))

        if not proprio_tensors:
            raise RuntimeError("No state keys found for LAM dataset.")
        proprio = torch.cat(proprio_tensors, dim=-1).contiguous()

        embodiment_id = int(dataset.embodiment_id)

        return {
            "frames": frames_t,
            "proprio": proprio,
            "embodiment_id": embodiment_id,
        }

    def __getitem__(self, index: int) -> Dict:
        # === Debug mode: cache and repeat samples ===
        if self.debug_repeat_batch:
            if not self._cache_initialized:
                # Initialize cache with k samples
                repeat_k = int(self.debug_repeat_batch) if isinstance(self.debug_repeat_batch, int) else 1
                repeat_k = max(1, repeat_k)
                print(f"[LeRobotLAMDataset] Debug mode: Caching {repeat_k} sample(s) for repeated use...")
                
                for i in range(repeat_k):
                    last_err: Optional[Exception] = None
                    for _ in range(self.max_retries):
                        try:
                            sample = self._try_get_sample(i)
                            self._cached_samples.append(sample)
                            break
                        except Exception as e:
                            last_err = e
                            continue
                    if last_err:
                        raise RuntimeError(f"Failed to cache sample {i} after retries: {last_err}")
                
                self._cache_initialized = True
                print(f"[LeRobotLAMDataset] Debug mode: Successfully cached {len(self._cached_samples)} sample(s).")
            
            # Return samples from cache in a round-robin manner
            cache_idx = index % len(self._cached_samples)
            return self._cached_samples[cache_idx]
        
        # === Normal mode ===
        last_err: Optional[Exception] = None
        for _ in range(self.max_retries):
            try:
                return self._try_get_sample(index)
            except Exception as e:  # retry on IO/video issues
                last_err = e
                index = random.randint(0, len(self.mixture) - 1)
                continue
        if last_err:
            raise last_err
        raise RuntimeError("Failed to fetch sample after retries.")
