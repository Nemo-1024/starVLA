from __future__ import annotations

from typing import Sequence

from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


def _normalize_state_keys(
    base_state_keys: Sequence[str],
    requested_state_keys: Sequence[str] | None,
) -> list[str]:
    if requested_state_keys is None:
        state_keys = list(base_state_keys)
    else:
        state_keys = list(requested_state_keys)

    if not state_keys:
        raise ValueError("No state keys provided for LAM state normalization.")

    for key in state_keys:
        if not key.startswith("state."):
            raise ValueError(f"Invalid state key '{key}'. Expected key prefix 'state.'.")

    base_set = set(base_state_keys)
    unknown_keys = [key for key in state_keys if key not in base_set]
    if unknown_keys:
        raise ValueError(
            f"Requested state keys are not present in base config: {unknown_keys}. "
            f"Available keys: {list(base_state_keys)}"
        )

    return state_keys


def filter_lam_video_keys(
    base_video_keys: Sequence[str],
    preferred_video_key: str | None = None,
) -> list[str]:
    """
    Keep all non-wrist camera keys for LAM.
    Single non-wrist view selection is handled downstream by mixture sampling
    policy.
    """
    filtered_video_keys = [key for key in base_video_keys if "wrist" not in key.lower()]
    if not filtered_video_keys:
        raise ValueError(
            "No non-wrist video keys available for LAM. "
            f"Received keys: {list(base_video_keys)}"
        )

    if preferred_video_key is not None:
        if preferred_video_key in filtered_video_keys:
            # Keep all non-wrist keys, but place preferred key first.
            return [preferred_video_key] + [
                key for key in filtered_video_keys if key != preferred_video_key
            ]
        if preferred_video_key in base_video_keys:
            raise ValueError(
                f"Preferred video key '{preferred_video_key}' is filtered out because it is a wrist view. "
                f"Available non-wrist keys: {filtered_video_keys}"
            )

    return filtered_video_keys


def build_lam_state_normalize_transform(
    robot_type: str,
    state_keys: Sequence[str] | None = None,
) -> ComposedModalityTransform:
    """
    Build a LAM-specific transform that only normalizes state keys.

    The transform is derived from the original robot config in ROBOT_TYPE_CONFIG_MAP,
    while stripping out video/action/concat/sin-cos related transforms.
    """
    if robot_type not in ROBOT_TYPE_CONFIG_MAP:
        raise ValueError(
            f"Unknown robot_type '{robot_type}'. "
            f"Known robot types: {list(ROBOT_TYPE_CONFIG_MAP.keys())}"
        )

    base_cfg = ROBOT_TYPE_CONFIG_MAP[robot_type]
    base_modality_cfg = base_cfg.modality_config()
    if "state" not in base_modality_cfg:
        raise ValueError(f"Robot type '{robot_type}' has no state modality config.")

    filtered_state_keys = _normalize_state_keys(
        base_state_keys=base_modality_cfg["state"].modality_keys,
        requested_state_keys=state_keys,
    )

    base_transform = base_cfg.transform()
    if not isinstance(base_transform, ComposedModalityTransform):
        raise TypeError(
            "Base data config transform must be ComposedModalityTransform, "
            f"got {type(base_transform)} for robot_type '{robot_type}'."
        )

    state_normalization_modes: dict[str, str] = {}
    state_tensor_input_dtypes = {}
    state_tensor_output_dtypes = {}
    found_state_action_transform = False

    for transform in base_transform.transforms:
        if isinstance(transform, StateActionToTensor):
            for key, dtype in transform.input_dtypes.items():
                if key in filtered_state_keys:
                    state_tensor_input_dtypes[key] = dtype
            for key, dtype in transform.output_dtypes.items():
                if key in filtered_state_keys:
                    state_tensor_output_dtypes[key] = dtype
        elif isinstance(transform, StateActionTransform):
            state_keys_in_transform = [key for key in transform.apply_to if key.startswith("state.")]
            if not state_keys_in_transform:
                continue
            found_state_action_transform = True
            for key, mode in transform.normalization_modes.items():
                if key.startswith("state."):
                    state_normalization_modes[key] = mode

    if not found_state_action_transform:
        raise ValueError(
            f"No state StateActionTransform found in base config for robot_type '{robot_type}'."
        )

    missing_modes = [key for key in filtered_state_keys if key not in state_normalization_modes]
    if missing_modes:
        raise ValueError(
            f"Missing normalization mode for state keys {missing_modes} "
            f"in robot_type '{robot_type}'."
        )

    filtered_modes = {key: state_normalization_modes[key] for key in filtered_state_keys}

    transforms = [
        StateActionToTensor(
            apply_to=filtered_state_keys,
            input_dtypes={
                key: state_tensor_input_dtypes[key]
                for key in filtered_state_keys
                if key in state_tensor_input_dtypes
            },
            output_dtypes={
                key: state_tensor_output_dtypes[key]
                for key in filtered_state_keys
                if key in state_tensor_output_dtypes
            },
        ),
        StateActionTransform(
            apply_to=filtered_state_keys,
            normalization_modes=filtered_modes,
            target_rotations={},
        ),
    ]
    return ComposedModalityTransform(transforms=transforms)
