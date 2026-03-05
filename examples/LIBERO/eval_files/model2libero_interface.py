from pathlib import Path
from typing import Dict, Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.tools import read_mode_config


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        image_size: list[int] = [256, 256],
        action_hz: float = 10.0,
        embodiment_id: int = 25,
        host: str = "127.0.0.1",
        port: int = 10095,
        **kwargs,   
    ) -> None:
        del kwargs
        if policy_ckpt_path is None or str(policy_ckpt_path).strip() == "":
            raise ValueError("`policy_ckpt_path` must be a non-empty checkpoint path.")
        self.policy_ckpt_path = Path(policy_ckpt_path).expanduser().resolve()

        self.model_config, self.norm_stats = read_mode_config(self.policy_ckpt_path)
        framework_name = self.model_config.get("framework", {}).get("name", "")
        if not self._is_latent_world_framework(framework_name):
            raise ValueError(
                "LIBERO online eval client now only supports latent_world_vla_independent. "
                f"Got framework.name={framework_name!r}."
            )
        action_cfg = self.model_config["framework"]["action_model"]
        self.enable_wrist_view = bool(action_cfg.get("enable_wrist_view", False))

        self.policy_setup = policy_setup
        self.unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        self.image_size = image_size
        self.action_hz = float(action_hz)
        if self.action_hz <= 0.0:
            raise ValueError(f"`action_hz` must be > 0, got {self.action_hz}.")
        self.embodiment_id = int(embodiment_id)

        self.action_norm_stats = self.norm_stats[self.unnorm_key]["action"]
        self.action_chunk_size = self.get_action_chunk_size(self.model_config)

        self.client = WebsocketClientPolicy(host, port)
        self._validate_server_metadata_or_raise(framework_name=framework_name)

        self.task_description = None
        self.raw_actions = None

        print(
            "*** "
            f"policy_setup: {policy_setup}, "
            f"unnorm_key: {self.unnorm_key}, "
            f"enable_wrist_view: {self.enable_wrist_view}, "
            f"action_hz: {self.action_hz} "
            "***"
        )

    @staticmethod
    def _normalize_framework_name(name: str) -> str:
        return str(name).replace("_", "").lower()

    @classmethod
    def _is_latent_world_framework(cls, framework_name: str) -> bool:
        return cls._normalize_framework_name(framework_name) == "latentworldvlaindependent"

    def _validate_server_metadata_or_raise(self, *, framework_name: str) -> None:
        metadata = self.client.get_server_metadata() or {}
        server_ckpt_raw = metadata.get("ckpt_path", None)
        if server_ckpt_raw is None:
            raise ValueError(
                "Server metadata does not contain `ckpt_path`; "
                "refuse to run because checkpoint consistency cannot be verified."
            )

        server_ckpt = Path(str(server_ckpt_raw)).expanduser().resolve()
        if server_ckpt != self.policy_ckpt_path:
            raise ValueError(
                "Checkpoint mismatch between eval client and server: "
                f"client_ckpt={self.policy_ckpt_path}, server_ckpt={server_ckpt}."
            )

        local_fw = self._normalize_framework_name(framework_name)
        if local_fw != "latentworldvlaindependent":
            raise ValueError(f"Unsupported local framework: {framework_name!r}.")

        server_fw_raw = metadata.get("framework_name", "")
        if server_fw_raw:
            server_fw = self._normalize_framework_name(server_fw_raw)
            if server_fw != "latentworldvlaindependent":
                raise ValueError(
                    "Server framework is not latent-world. "
                    f"server_framework={server_fw_raw!r}."
                )

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.raw_actions = None

    def step(self, example: dict, step: int = 0, **kwargs) -> dict[str, np.ndarray]:
        del kwargs
        task_description = example.get("lang", None)
        if task_description is None:
            raise KeyError("LIBERO eval example must contain key `lang`.")
        primary_images = example.get("primary_image", None)
        if not isinstance(primary_images, (list, tuple)) or len(primary_images) == 0:
            raise ValueError("LIBERO online infer requires `example['primary_image']` as a non-empty list.")
        if len(primary_images) != 1:
            raise ValueError(
                "LIBERO online infer expects exactly one primary view in `example['primary_image']`, "
                f"got {len(primary_images)}."
            )
        if not isinstance(primary_images[0], np.ndarray):
            raise TypeError(
                "LIBERO online infer requires `example['primary_image'][0]` to be `np.ndarray`, "
                f"got {type(primary_images[0]).__name__}."
            )

        state = example.get("state", None)
        if state is None:
            raise KeyError("LIBERO eval example must contain key `state`.")
        state_arr = np.asarray(state, dtype=np.float32)
        if state_arr.ndim == 1:
            state_arr = state_arr[None, :]
        if state_arr.ndim != 2:
            raise ValueError(f"`state` must have shape [D] or [T,D], got {state_arr.shape}.")

        if task_description != self.task_description:
            self.reset(task_description)

        primary = self._resize_image(np.asarray(primary_images[0]))
        wrist_raw = example.get("wrist_image", None)

        inference_example = {
            "primary_image": [primary],
            "lang": str(task_description),
            "state": state_arr,
            "embodiment_id": int(example.get("embodiment_id", self.embodiment_id)),
            "action_hz": float(example.get("action_hz", self.action_hz)),
        }

        if inference_example["action_hz"] <= 0.0:
            raise ValueError(f"`action_hz` must be > 0, got {inference_example['action_hz']}.")

        if self.enable_wrist_view:
            if not isinstance(wrist_raw, (list, tuple)) or len(wrist_raw) == 0:
                raise ValueError(
                    "Checkpoint requires wrist view (`enable_wrist_view=true`), "
                    "but no `wrist_image` was provided."
                )
            if not all(isinstance(image, np.ndarray) for image in wrist_raw):
                raise TypeError("`wrist_image` entries must all be `np.ndarray`.")
            inference_example["wrist_image"] = [
                self._resize_image(np.asarray(image)) for image in wrist_raw
            ]
        elif wrist_raw is not None:
            if not isinstance(wrist_raw, (list, tuple)):
                raise ValueError("`wrist_image` must be a list/tuple when provided.")
            if not all(isinstance(image, np.ndarray) for image in wrist_raw):
                raise TypeError("`wrist_image` entries must all be `np.ndarray`.")
            inference_example["wrist_image"] = [
                self._resize_image(np.asarray(image)) for image in wrist_raw
            ]

        vla_input = {"examples": [inference_example]}

        if step % self.action_chunk_size == 0 or self.raw_actions is None:
            response = self.client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"]
            except KeyError:
                raise KeyError(
                    f"Key 'normalized_actions' not found in response: {response}"
                )

            normalized_actions = normalized_actions[0]
            self.raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions,
                action_norm_stats=self.action_norm_stats,
            )

        raw_actions = self.raw_actions[step % self.action_chunk_size][None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),
        }
        return {"raw_action": raw_action}

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        high_key = "max" if "max" in action_norm_stats else "q99"
        low_key = "min" if "min" in action_norm_stats else "q01"
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats[low_key], dtype=bool))
        action_high = np.asarray(action_norm_stats[high_key], dtype=np.float32)
        action_low = np.asarray(action_norm_stats[low_key], dtype=np.float32)
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        return np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

    @staticmethod
    def get_action_chunk_size(model_config: dict) -> int:
        action_cfg = model_config["framework"]["action_model"]
        if action_cfg.get("action_horizon", None) is not None:
            return int(action_cfg["action_horizon"])
        future = int(action_cfg.get("future_action_window_size", 0))
        past = int(action_cfg.get("past_action_window_size", 0))
        if "future_action_window_size" in action_cfg:
            return future + past + 1
        raise KeyError(
            "Cannot infer action chunk size from config; missing `action_horizon` and "
            "`future_action_window_size`."
        )

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            if len(norm_stats) != 1:
                raise ValueError(
                    "Checkpoint contains multiple dataset statistics; please pass `unnorm_key`. "
                    f"Available: {list(norm_stats.keys())}"
                )
            return next(iter(norm_stats.keys()))
        if unnorm_key not in norm_stats:
            raise ValueError(
                f"Invalid `unnorm_key`={unnorm_key}. Available: {list(norm_stats.keys())}"
            )
        return unnorm_key
