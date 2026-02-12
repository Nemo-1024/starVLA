import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.data_config_lam import (
    build_lam_state_normalize_transform,
)
from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
    ModalityConfig,
)
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG
from starVLA.dataloader.gr00t_lerobot.video import DECORD_AVAILABLE, TORCHCODEC_AVAILABLE


def _build_modality_config(
    robot_type: str,
    num_frames: int,
    preferred_video_key: Optional[str] = None,
    state_keys: Optional[Sequence[str]] = None,
    frame_stride: int = 1,
    video_delta_indices: Optional[Sequence[int]] = None,
    state_delta_indices: Optional[Sequence[int]] = None,
    include_action: bool = False,
    include_language: bool = False,
) -> Dict[str, ModalityConfig]:
    """
    Clone ROBOT_TYPE_CONFIG_MAP[robot_type].modality_config() and override delta_indices.
    We only keep video/state by default to reduce IO for LAM.
    """
    base_cfg = ROBOT_TYPE_CONFIG_MAP[robot_type].modality_config()

    def _default_delta_indices() -> list[int]:
        if frame_stride <= 0:
            raise ValueError(f"frame_stride must be a positive int, got {frame_stride}")
        # Example: num_frames=5, frame_stride=2 -> [0,2,4,6,8]
        return list(range(0, num_frames * frame_stride, frame_stride))

    def override_delta(
        cfg: ModalityConfig,
        keys: Optional[Sequence[str]],
        delta_indices: Optional[Sequence[int]] = None,
    ) -> ModalityConfig:
        new_keys = list(keys) if keys is not None else cfg.modality_keys
        if delta_indices is None:
            delta_list = _default_delta_indices()
        else:
            delta_list = list(delta_indices)
            if len(delta_list) == 0:
                raise ValueError("delta_indices must be non-empty if provided.")
        return ModalityConfig(delta_indices=delta_list, modality_keys=new_keys)

    # video
    video_keys = base_cfg["video"].modality_keys
    if preferred_video_key and preferred_video_key in video_keys:
        video_keys = [preferred_video_key]
    video_modality = override_delta(base_cfg["video"], video_keys, video_delta_indices)

    # state
    state_modality = override_delta(base_cfg["state"], state_keys, state_delta_indices)

    modality_configs: Dict[str, ModalityConfig] = {
        "video": video_modality,
        "state": state_modality,
    }

    if include_action and "action" in base_cfg:
        modality_configs["action"] = override_delta(base_cfg["action"], base_cfg["action"].modality_keys)
    if include_language and "language" in base_cfg:
        modality_configs["language"] = base_cfg["language"]

    return modality_configs


class LeRobotLAMDataset(Dataset):
    """
    Mixture dataset that reuses starVLA's LeRobot loaders but emits LAM-ready raw samples.

    Output sample:
        {
            "frames": torch.Tensor[T, H, W, C] uint8,
            "proprio": torch.Tensor[T, D] float32,
            "dataset_id": int,
        }
    """

    def __init__(
        self,
        data_root_dir: str | Path,
        data_mix: str,
        num_frames: int,
        video_backend: str = "torchvision_av",
        preferred_video_key: Optional[str] = None,
        state_keys: Optional[Sequence[str]] = None,
        frame_dt_sec: Optional[float] = None,
        frame_stride: int = 1,
        video_delta_indices: Optional[Sequence[int]] = None,
        state_delta_indices: Optional[Sequence[int]] = None,
        max_retries: int = 5,
        debug_repeat_batch: Union[bool, int] = False,
    ) -> None:
        super().__init__()
        self.data_root_dir = Path(data_root_dir)
        self.data_mix = data_mix
        self.num_frames = num_frames
        self.preferred_video_key = preferred_video_key
        self.state_keys = list(state_keys) if state_keys else None
        self.frame_dt_sec = frame_dt_sec
        self.frame_stride = frame_stride
        self.video_delta_indices = list(video_delta_indices) if video_delta_indices is not None else None
        self.state_delta_indices = list(state_delta_indices) if state_delta_indices is not None else None
        self.max_retries = max_retries

        # "auto" 选择在不同环境里更稳健：
        # - 优先 decord（通常最快）
        # - 其次 torchcodec
        # - 否则回退到 torchvision_av
        if video_backend == "auto":
            if DECORD_AVAILABLE:
                video_backend = "decord"
            elif TORCHCODEC_AVAILABLE:
                video_backend = "torchcodec"
            else:
                video_backend = "torchvision_av"
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
        self.dataset_id_mapping: Dict[int, str] = {}
        self._dataset_object_to_id: Dict[int, int] = {}

        for dataset_name, weight, robot_type in mixture_spec:
            key = (dataset_name, robot_type)
            if key in seen:
                continue
            seen.add(key)

            modality_cfg = _build_modality_config(
                robot_type=robot_type,
                num_frames=self.num_frames,
                frame_stride=self.frame_stride,
                video_delta_indices=self.video_delta_indices,
                state_delta_indices=self.state_delta_indices,
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
            new_idx = len(dataset_mixture)
            dataset_mixture.append((ds, weight))
            self.dataset_id_mapping[new_idx] = dataset_name
            self._dataset_object_to_id[id(ds)] = new_idx

        self.mixture = LeRobotMixtureDataset(
            dataset_mixture,
            mode="train",
            balance_dataset_weights=True,
            seed=42,
            data_cfg=data_cfg,
        )

    def _maybe_override_delta_indices_by_dt(self, dataset: LeRobotSingleDataset) -> None:
        """
        If frame_dt_sec is set (and explicit delta indices are not provided), override
        video/state delta_indices to approximate a constant time interval in seconds,
        using the dataset video fps.

        Note: this assumes the trajectory "step" timestamps align reasonably with video timestamps.
        """
        if self.frame_dt_sec is None:
            return
        if self.video_delta_indices is not None or self.state_delta_indices is not None:
            return

        if self.frame_dt_sec <= 0:
            raise ValueError(f"frame_dt_sec must be > 0, got {self.frame_dt_sec}")

        if not dataset.modality_keys.get("video"):
            return

        # Pick the first configured video key (LAM usually narrows to 1 via preferred_video_key).
        video_key_full = dataset.modality_keys["video"][0]
        video_subkey = video_key_full.replace("video.", "")
        fps = float(dataset.metadata.modalities.video[video_subkey].fps)
        stride = max(1, int(round(self.frame_dt_sec * fps)))

        delta_list = list(range(0, self.num_frames * stride, stride))
        delta_arr = np.array(delta_list, dtype=np.int64)

        # Override in-place for all configured video/state keys in this dataset instance.
        for k in dataset.modality_keys.get("video", []):
            dataset._delta_indices[k] = delta_arr
        for k in dataset.modality_keys.get("state", []):
            dataset._delta_indices[k] = delta_arr

    def __len__(self) -> int:
        # In debug mode, return a large virtual length to support multiple epochs
        if self.debug_repeat_batch:
            return 10000
        return len(self.mixture)

    def _try_get_sample(self, index: int) -> Dict:
        dataset, traj_id, base_index = self.mixture.sample_step(index)
        self._maybe_override_delta_indices_by_dt(dataset)
        raw_data = dataset.get_step_data(traj_id, base_index)
        data = dataset.transforms(raw_data)

        # video
        available_video_keys = dataset.modality_keys["video"]
        if self.preferred_video_key and self.preferred_video_key in available_video_keys:
            video_key = self.preferred_video_key
        else:
            video_key = available_video_keys[0]
        frames = data[video_key]  # (T, H, W, C)
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

        dataset_id = self._dataset_object_to_id.get(id(dataset), 0)

        return {
            "frames": frames_t,
            "proprio": proprio,
            "dataset_id": dataset_id,
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
