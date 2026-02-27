from collections import deque
from typing import Optional, Sequence
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
from typing import Dict
from pathlib import Path
from PIL import Image

from starVLA.model.tools import read_mode_config

class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble = True,
        action_ensemble_horizon: Optional[int] = 3, # different cross sim
        image_size: list[int] = [256, 256],
        adaptive_ensemble_alpha = 0.1,
        host="0.0.0.0",
        port=10095,
    ) -> None:
        if policy_ckpt_path is None or str(policy_ckpt_path).strip() == "":
            raise ValueError("`policy_ckpt_path` must be a non-empty checkpoint path.")
        self.policy_ckpt_path = Path(policy_ckpt_path).expanduser().resolve()

        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)
        self._validate_server_metadata_or_raise()
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.image_size = image_size
        self.horizon = horizon #0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=self.policy_ckpt_path)
        self.state_norm_stats = self.get_state_stats(self.unnorm_key, policy_ckpt_path=self.policy_ckpt_path)
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=self.policy_ckpt_path)
        self.raw_actions = None

    def _validate_server_metadata_or_raise(self) -> None:
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

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.raw_actions = None


    def step(
        self, 
        example: dict,
        step: int = 0,
        **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Perform one step of inference
        :param image: Input image in the format (H, W, 3), type uint8
        :param task_description: Task description text
        :return: (raw action, processed action)
        """

        task_description = example.get("lang", None)
        images = example["image"]  # list of images for history
        state = example.get("state", None)
        if state is None:
            raise KeyError("LIBERO eval example must contain key `state`.")

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        images = [self._resize_image(image) for image in images]
        inference_example = dict(example)
        inference_example["image"] = images
        if "wrist_images" in inference_example and inference_example["wrist_images"] is not None:
            inference_example["wrist_images"] = [
                self._resize_image(image) for image in inference_example["wrist_images"]
            ]
        inference_example["state"] = self.normalize_state(state, self.state_norm_stats)
        inference_example["embodiment_id"] = 25
        vla_input = {
            "examples": [inference_example],
        }

        action_chunk_size = self.action_chunk_size
        if step % action_chunk_size == 0 or self.raw_actions is None:
            response = self.client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"]  # B, chunk, D
            except KeyError:
                print(f"Response data: {response}")
                raise KeyError(f"Key 'normalized_actions' not found in response data: {response['data'].keys()}")

            normalized_actions = normalized_actions[0]
            self.raw_actions = self.unnormalize_actions(normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats)

        raw_actions = self.raw_actions[step % action_chunk_size][None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        return {"raw_action": raw_action}

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        high_key = "max" if "max" in action_norm_stats else "q99"
        low_key = "min" if "min" in action_norm_stats else "q01"
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats[low_key], dtype=bool))
        action_high, action_low = np.array(action_norm_stats[high_key]), np.array(action_norm_stats[low_key])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_state_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        _, norm_stats = read_mode_config(policy_ckpt_path)

        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["state"]

    @staticmethod
    def normalize_state(state: np.ndarray, state_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        state_arr = np.asarray(state, dtype=np.float32)
        if state_arr.ndim == 1:
            state_arr = state_arr[None, :]
        if state_arr.ndim != 2:
            raise ValueError(
                f"`state` must have shape [D] or [T, D], got shape={state_arr.shape}."
            )

        state_min = np.asarray(state_norm_stats["min"], dtype=np.float32).reshape(-1)
        state_max = np.asarray(state_norm_stats["max"], dtype=np.float32).reshape(-1)
        if state_arr.shape[-1] == state_min.shape[0] + 1:
            # LIBERO 8-dim state -> keep first 6 dims and the last gripper dim.
            state_arr = np.concatenate(
                [state_arr[:, : state_min.shape[0] - 1], state_arr[:, -1:]],
                axis=1,
            )
        if state_arr.shape[-1] != state_min.shape[0]:
            raise ValueError(
                f"State dim mismatch: state_dim={state_arr.shape[-1]} vs stats_dim={state_min.shape[0]}."
            )

        denom = state_max - state_min
        valid = denom != 0
        normalized = np.zeros_like(state_arr, dtype=np.float32)
        normalized[:, valid] = (
            (state_arr[:, valid] - state_min[valid]) / denom[valid]
        ) * 2.0 - 1.0

        # Keep LIBERO gripper state binary to match training transform contract.
        normalized[:, -1] = (state_arr[:, -1] > 0.5).astype(np.float32)
        return normalized

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)  # read config and norm_stats
        action_cfg = model_config["framework"]["action_model"]
        if action_cfg.get("action_horizon", None) is not None:
            return int(action_cfg["action_horizon"])
        future = int(action_cfg.get("future_action_window_size", 0))
        past = int(action_cfg.get("past_action_window_size", 0))
        if "future_action_window_size" in action_cfg:
            return future + past + 1
        raise KeyError("Cannot infer action chunk size from config; missing `action_horizon` and `future_action_window_size`.")


    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
    
    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
