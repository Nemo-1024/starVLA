#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R


def _ensure_lerobot_on_path() -> None:
    if importlib.util.find_spec("lerobot") is not None:
        return

    candidates = [
        Path.cwd() / ".." / "lerobot" / "src",
        Path(__file__).resolve().parents[2] / "lerobot" / "src",
        Path("/mnt/project_rlinf/jlchen/code/lerobot/src"),
    ]

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            if importlib.util.find_spec("lerobot") is not None:
                return

    raise ModuleNotFoundError(
        "Cannot import 'lerobot'. Please ensure lerobot is installed or add its src path to PYTHONPATH."
    )


def _ensure_starvla_on_path() -> None:
    if importlib.util.find_spec("starVLA") is not None:
        return
    repo_root = Path(__file__).resolve().parents[1]
    if repo_root.exists():
        sys.path.insert(0, str(repo_root))
    if importlib.util.find_spec("starVLA") is None:
        raise ModuleNotFoundError(
            "Cannot import 'starVLA'. Please run from repo root or add starVLA package path to PYTHONPATH."
        )


_ensure_lerobot_on_path()
_ensure_starvla_on_path()

from lerobot.datasets.compute_stats import DEFAULT_QUANTILES, aggregate_stats, get_feature_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_stats, write_stats


OBS_LEFT_EEF_KEY = "observation.states.end_effector_left"
OBS_RIGHT_EEF_KEY = "observation.states.end_effector_right"
OBS_LEFT_JOINT_POS_KEY = "observation.states.joint_position_left"
OBS_RIGHT_JOINT_POS_KEY = "observation.states.joint_position_right"
EPISODE_KEY = "episode_index"
DEFAULT_LEFT_STATE_ROTVEC_KEY = "observation.states.end_effector_left_rotvec"
DEFAULT_RIGHT_STATE_ROTVEC_KEY = "observation.states.end_effector_right_rotvec"


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def _collect_data_files(root: Path) -> list[Path]:
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")
    files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}/chunk-*/file-*.parquet")
    return files


def _to_2d_array(column_values: pd.Series, key: str) -> np.ndarray:
    rows = []
    for value in column_values:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        rows.append(arr.reshape(-1))
    try:
        return np.stack(rows, axis=0)
    except Exception as exc:
        raise ValueError(f"Failed to stack values for key '{key}' into a consistent array.") from exc


def _extract_state_and_episode_arrays(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for key in (
        OBS_LEFT_EEF_KEY,
        OBS_RIGHT_EEF_KEY,
        OBS_LEFT_JOINT_POS_KEY,
        OBS_RIGHT_JOINT_POS_KEY,
        EPISODE_KEY,
    ):
        if key not in df.columns:
            raise KeyError(f"Required column missing in parquet: {key}")

    left = _to_2d_array(df[OBS_LEFT_EEF_KEY], OBS_LEFT_EEF_KEY)
    right = _to_2d_array(df[OBS_RIGHT_EEF_KEY], OBS_RIGHT_EEF_KEY)
    if left.shape[1] != 7 or right.shape[1] != 7:
        raise ValueError(
            f"Expected EEF dim=7 for both arms. left={left.shape}, right={right.shape}."
        )

    left_joint = _to_2d_array(df[OBS_LEFT_JOINT_POS_KEY], OBS_LEFT_JOINT_POS_KEY)
    right_joint = _to_2d_array(df[OBS_RIGHT_JOINT_POS_KEY], OBS_RIGHT_JOINT_POS_KEY)
    if left_joint.shape[1] < 1 or right_joint.shape[1] < 1:
        raise ValueError(
            f"Expected joint_position dim>=1 for both arms. left={left_joint.shape}, right={right_joint.shape}."
        )
    left_gripper = left_joint[:, -1].astype(np.float32)
    right_gripper = right_joint[:, -1].astype(np.float32)

    episodes = df[EPISODE_KEY].to_numpy()
    if episodes.ndim != 1:
        episodes = episodes.reshape(-1)
    return left, right, left_gripper, right_gripper, episodes.astype(np.int64)


def _read_first_row_info(file_path: Path) -> tuple[np.ndarray, np.ndarray, float, float, int] | None:
    small_df = pd.read_parquet(
        file_path,
        columns=[
            OBS_LEFT_EEF_KEY,
            OBS_RIGHT_EEF_KEY,
            OBS_LEFT_JOINT_POS_KEY,
            OBS_RIGHT_JOINT_POS_KEY,
            EPISODE_KEY,
        ],
    )
    if len(small_df) == 0:
        return None
    left = np.asarray(small_df[OBS_LEFT_EEF_KEY].iloc[0], dtype=np.float32).reshape(-1)
    right = np.asarray(small_df[OBS_RIGHT_EEF_KEY].iloc[0], dtype=np.float32).reshape(-1)
    if left.shape[0] != 7 or right.shape[0] != 7:
        raise ValueError(
            f"Invalid first-row EEF dim in {file_path}. left={left.shape}, right={right.shape}."
        )
    left_joint = np.asarray(small_df[OBS_LEFT_JOINT_POS_KEY].iloc[0], dtype=np.float32).reshape(-1)
    right_joint = np.asarray(small_df[OBS_RIGHT_JOINT_POS_KEY].iloc[0], dtype=np.float32).reshape(-1)
    if left_joint.shape[0] < 1 or right_joint.shape[0] < 1:
        raise ValueError(
            f"Invalid first-row joint_position dim in {file_path}. "
            f"left={left_joint.shape}, right={right_joint.shape}."
        )
    left_gripper = float(left_joint[-1])
    right_gripper = float(right_joint[-1])
    ep = int(small_df[EPISODE_KEY].iloc[0])
    return left, right, left_gripper, right_gripper, ep


def _normalize_quaternions_with_mask(quat_xyzw: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(quat_xyzw).all(axis=1)
    norms = np.linalg.norm(quat_xyzw, axis=1)
    valid = finite & (norms > eps)
    normalized = np.zeros_like(quat_xyzw, dtype=np.float32)
    if np.any(valid):
        normalized[valid] = quat_xyzw[valid] / norms[valid, None]
    return normalized, valid


def _compute_delta_from_pairs(
    eef_t: np.ndarray,
    eef_t1: np.ndarray,
    valid_pair_mask: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, int]:
    num_rows = eef_t.shape[0]
    out = np.zeros((num_rows, 6), dtype=np.float32)
    if num_rows == 0 or not np.any(valid_pair_mask):
        return out, 0

    valid_rows = np.where(valid_pair_mask)[0]
    t = eef_t[valid_rows]
    t1 = eef_t1[valid_rows]

    out[valid_rows, :3] = t1[:, :3] - t[:, :3]

    q_t, q_t_ok = _normalize_quaternions_with_mask(t[:, 3:7], eps=eps)
    q_t1, q_t1_ok = _normalize_quaternions_with_mask(t1[:, 3:7], eps=eps)
    rot_ok = q_t_ok & q_t1_ok

    bad_count = int((~rot_ok).sum())
    if np.any(rot_ok):
        delta_rotvec = (R.from_quat(q_t1[rot_ok]) * R.from_quat(q_t[rot_ok]).inv()).as_rotvec().astype(np.float32)
        out[valid_rows[rot_ok], 3:6] = delta_rotvec

    return out, bad_count


def _compute_gripper_delta_from_pairs(
    gripper_t: np.ndarray,
    gripper_t1: np.ndarray,
    valid_pair_mask: np.ndarray,
) -> np.ndarray:
    out = np.zeros((gripper_t.shape[0], 1), dtype=np.float32)
    if gripper_t.shape[0] == 0 or not np.any(valid_pair_mask):
        return out
    valid_rows = np.where(valid_pair_mask)[0]
    out[valid_rows, 0] = gripper_t1[valid_rows] - gripper_t[valid_rows]
    return out


def _compute_state_rotvec_from_eef(eef_state: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, int]:
    num_rows = eef_state.shape[0]
    out = np.zeros((num_rows, 3), dtype=np.float32)
    if num_rows == 0:
        return out, 0

    q_state, q_ok = _normalize_quaternions_with_mask(eef_state[:, 3:7], eps=eps)
    bad_count = int((~q_ok).sum())
    if np.any(q_ok):
        out[q_ok] = R.from_quat(q_state[q_ok]).as_rotvec().astype(np.float32)
    return out, bad_count


def _compute_delta_columns_for_file(
    left_curr: np.ndarray,
    right_curr: np.ndarray,
    left_gripper_curr: np.ndarray,
    right_gripper_curr: np.ndarray,
    ep_curr: np.ndarray,
    next_first_row: tuple[np.ndarray, np.ndarray, float, float, int] | None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    num_rows = left_curr.shape[0]
    if num_rows == 0:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0, 7), dtype=np.float32), 0, 0

    left_next = np.zeros_like(left_curr, dtype=np.float32)
    right_next = np.zeros_like(right_curr, dtype=np.float32)
    left_gripper_next = np.zeros_like(left_gripper_curr, dtype=np.float32)
    right_gripper_next = np.zeros_like(right_gripper_curr, dtype=np.float32)
    ep_next = np.full((num_rows,), -1, dtype=np.int64)

    if num_rows > 1:
        left_next[:-1] = left_curr[1:]
        right_next[:-1] = right_curr[1:]
        left_gripper_next[:-1] = left_gripper_curr[1:]
        right_gripper_next[:-1] = right_gripper_curr[1:]
        ep_next[:-1] = ep_curr[1:]

    if next_first_row is not None:
        left_next[-1] = next_first_row[0]
        right_next[-1] = next_first_row[1]
        left_gripper_next[-1] = next_first_row[2]
        right_gripper_next[-1] = next_first_row[3]
        ep_next[-1] = next_first_row[4]

    valid_pairs = ep_curr == ep_next

    left_eef_delta, left_bad = _compute_delta_from_pairs(left_curr, left_next, valid_pairs)
    right_eef_delta, right_bad = _compute_delta_from_pairs(right_curr, right_next, valid_pairs)
    left_gripper_delta = _compute_gripper_delta_from_pairs(left_gripper_curr, left_gripper_next, valid_pairs)
    right_gripper_delta = _compute_gripper_delta_from_pairs(right_gripper_curr, right_gripper_next, valid_pairs)

    left_delta = np.concatenate([left_eef_delta, left_gripper_delta], axis=1)
    right_delta = np.concatenate([right_eef_delta, right_gripper_delta], axis=1)
    return left_delta.astype(np.float32), right_delta.astype(np.float32), left_bad, right_bad


def _feature_names() -> list[str]:
    return ["dx", "dy", "dz", "rx", "ry", "rz", "dg"]


def _update_info_json(
    output_root: Path,
    left_key: str,
    right_key: str,
    left_state_rotvec_key: str,
    right_state_rotvec_key: str,
    add_action_output: bool,
    add_state_output: bool,
) -> None:
    info_path = output_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    info = _load_json(info_path)
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Invalid info.json features in {info_path}")

    if add_action_output:
        existing = [k for k in (left_key, right_key) if k in features]
        if existing:
            raise ValueError(f"Output key(s) already exist in info.json features: {existing}")
        feature_value = {"dtype": "float32", "shape": [7], "names": {"motors": _feature_names()}}
        features[left_key] = dict(feature_value)
        features[right_key] = dict(feature_value)

    if add_state_output:
        existing_state = [k for k in (left_state_rotvec_key, right_state_rotvec_key) if k in features]
        if existing_state:
            raise ValueError(f"State rotvec key(s) already exist in info.json features: {existing_state}")
        state_feature = {"dtype": "float32", "shape": [3], "names": {"motors": ["rx", "ry", "rz"]}}
        features[left_state_rotvec_key] = dict(state_feature)
        features[right_state_rotvec_key] = dict(state_feature)

    info["features"] = features
    _save_json(info_path, info)


def _update_modality_json(
    output_root: Path,
    left_key: str,
    right_key: str,
    left_state_rotvec_key: str,
    right_state_rotvec_key: str,
    add_action_output: bool,
    add_state_output: bool,
) -> None:
    modality_path = output_root / "meta" / "modality.json"
    if not modality_path.exists():
        raise FileNotFoundError(f"Missing modality.json: {modality_path}")

    modality = _load_json(modality_path)
    if add_action_output:
        if "action" not in modality or not isinstance(modality["action"], dict):
            raise ValueError("modality.json must contain top-level 'action' dict.")

        action_node = modality["action"]
        new_keys = [
            "delta_left_eef_position",
            "delta_left_eef_orientation",
            "delta_left_gripper",
            "delta_right_eef_position",
            "delta_right_eef_orientation",
            "delta_right_gripper",
        ]
        duplicate_keys = [k for k in new_keys if k in action_node]
        if duplicate_keys:
            raise ValueError(f"modality.json action keys already exist: {duplicate_keys}")

        action_node["delta_left_eef_position"] = {
            "start": 0,
            "end": 3,
            "original_key": left_key,
            "absolute": False,
            "dtype": "float32",
        }
        action_node["delta_left_eef_orientation"] = {
            "start": 3,
            "end": 6,
            "rotation_type": "axis_angle",
            "original_key": left_key,
            "absolute": False,
            "dtype": "float32",
        }
        action_node["delta_left_gripper"] = {
            "start": 6,
            "end": 7,
            "original_key": left_key,
            "absolute": False,
            "dtype": "float32",
        }
        action_node["delta_right_eef_position"] = {
            "start": 0,
            "end": 3,
            "original_key": right_key,
            "absolute": False,
            "dtype": "float32",
        }
        action_node["delta_right_eef_orientation"] = {
            "start": 3,
            "end": 6,
            "rotation_type": "axis_angle",
            "original_key": right_key,
            "absolute": False,
            "dtype": "float32",
        }
        action_node["delta_right_gripper"] = {
            "start": 6,
            "end": 7,
            "original_key": right_key,
            "absolute": False,
            "dtype": "float32",
        }

    if add_state_output:
        if "state" not in modality or not isinstance(modality["state"], dict):
            raise ValueError("modality.json must contain top-level 'state' dict.")
        state_node = modality["state"]
        state_keys = ["left_eef_orientation_rotvec", "right_eef_orientation_rotvec"]
        duplicate_state_keys = [k for k in state_keys if k in state_node]
        if duplicate_state_keys:
            raise ValueError(f"modality.json state keys already exist: {duplicate_state_keys}")
        state_node["left_eef_orientation_rotvec"] = {
            "start": 0,
            "end": 3,
            "rotation_type": "axis_angle",
            "original_key": left_state_rotvec_key,
            "absolute": True,
            "dtype": "float32",
        }
        state_node["right_eef_orientation_rotvec"] = {
            "start": 0,
            "end": 3,
            "rotation_type": "axis_angle",
            "original_key": right_state_rotvec_key,
            "absolute": True,
            "dtype": "float32",
        }

    _save_json(modality_path, modality)


def _to_tensor_list(values: list[Any], key: str) -> torch.Tensor:
    tensors = []
    for value in values:
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu()
        else:
            try:
                t = torch.as_tensor(value)
            except Exception:
                t = torch.as_tensor(np.asarray(value))
        tensors.append(t)
    try:
        return torch.stack(tensors)
    except Exception as exc:
        raise ValueError(f"Failed to stack feature '{key}' for stats computation.") from exc


def _process_single_episode_for_stats(
    dataset: LeRobotDataset,
    episode_idx: int,
    include_keys: set[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    start_idx = int(dataset.meta.episodes[episode_idx]["dataset_from_index"])
    end_idx = int(dataset.meta.episodes[episode_idx]["dataset_to_index"])

    all_feature_keys = set(dataset.features.keys())
    if include_keys is None:
        target_keys = {k for k, ft in dataset.features.items() if ft["dtype"] != "string"}
    else:
        missing = include_keys - all_feature_keys
        if missing:
            raise ValueError(f"Requested stats keys are not present in dataset features: {sorted(missing)}")
        target_keys = {k for k in include_keys if dataset.features[k]["dtype"] != "string"}

    if not target_keys:
        return {}

    episode_ds = dataset.hf_dataset.select_columns(sorted(target_keys)).select(range(start_idx, end_idx))
    collected_data: dict[str, list[Any]] = {k: [] for k in target_keys}
    for row in episode_ds:
        for key in target_keys:
            collected_data[key].append(row[key])

    episode_stats: dict[str, dict[str, np.ndarray]] = {}
    for key, values in collected_data.items():
        dtype = dataset.features[key]["dtype"]
        if dtype == "string":
            continue

        data = _to_tensor_list(values, key).cpu().numpy()
        if dtype in ["image", "video"]:
            if data.dtype == np.uint8:
                data = data.astype(np.float32) / 255.0
            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            axes_to_reduce = 0
            keepdims = data.ndim == 1

        stats = get_feature_stats(
            data,
            axis=axes_to_reduce,
            keepdims=keepdims,
            quantile_list=DEFAULT_QUANTILES,
        )
        if dtype in ["image", "video"]:
            stats = {k: v if k == "count" else np.squeeze(v, axis=0) for k, v in stats.items()}
        episode_stats[key] = stats
    return episode_stats


def _recompute_stats(
    output_root: Path,
    target_keys: set[str] | None = None,
    append_to_existing: bool = True,
) -> None:
    dataset = LeRobotDataset(repo_id=output_root.name, root=output_root, video_backend="pyav")
    if target_keys is None:
        target_keys = {k for k, ft in dataset.features.items() if ft["dtype"] != "string"}

    episode_stats_list = []
    for episode_idx in range(dataset.num_episodes):
        ep_stats = _process_single_episode_for_stats(dataset, episode_idx, include_keys=target_keys)
        if ep_stats:
            episode_stats_list.append(ep_stats)

    if not episode_stats_list:
        raise ValueError("No episode statistics were collected.")

    merged_stats = aggregate_stats(episode_stats_list)
    expected_feature_keys = set(target_keys)
    missing_keys = expected_feature_keys - set(merged_stats.keys())
    if missing_keys:
        raise ValueError(
            "Stats recompute did not cover all non-string features. "
            f"Missing keys: {sorted(missing_keys)}"
        )

    if append_to_existing:
        existing_stats = load_stats(output_root)
        merged_with_existing = {} if existing_stats is None else dict(existing_stats)
        merged_with_existing.update({k: merged_stats[k] for k in expected_feature_keys})
        write_stats(merged_with_existing, output_root)
    else:
        write_stats(merged_stats, output_root)


def _prepare_output_dir(dataset_root: Path, output_root: Path, overwrite_output: bool, in_place: bool) -> None:
    if in_place:
        return
    if output_root.exists():
        if not overwrite_output:
            raise FileExistsError(f"Output dir already exists: {output_root}. Use --overwrite-output to replace it.")
        shutil.rmtree(output_root)
    shutil.copytree(dataset_root, output_root)


def _process_data_files(
    output_root: Path,
    left_output_key: str,
    right_output_key: str,
    left_state_rotvec_key: str,
    right_state_rotvec_key: str,
    dry_run: bool,
    add_action_output: bool,
    add_state_output: bool,
) -> tuple[int | None, int | None, int, int, int | None, int | None, int, int]:
    files = _collect_data_files(output_root)
    if not files:
        raise ValueError("No parquet files found.")

    first_rows = [_read_first_row_info(fp) for fp in files]

    left_dim = -1 if add_action_output else None
    right_dim = -1 if add_action_output else None
    left_bad_total = 0
    right_bad_total = 0
    left_state_dim = -1 if add_state_output else None
    right_state_dim = -1 if add_state_output else None
    left_state_bad_total = 0
    right_state_bad_total = 0

    for idx, file_path in enumerate(files):
        df = pd.read_parquet(file_path)
        row_count = len(df)
        if row_count == 0:
            continue

        if add_action_output and (left_output_key in df.columns or right_output_key in df.columns):
            raise ValueError(
                f"Output key already exists in file {file_path}. "
                f"left_exists={left_output_key in df.columns}, right_exists={right_output_key in df.columns}"
            )
        if add_state_output and (left_state_rotvec_key in df.columns or right_state_rotvec_key in df.columns):
            raise ValueError(
                f"State rotvec key already exists in file {file_path}. "
                f"left_exists={left_state_rotvec_key in df.columns}, right_exists={right_state_rotvec_key in df.columns}"
            )

        left_curr, right_curr, left_gripper_curr, right_gripper_curr, ep_curr = _extract_state_and_episode_arrays(df)
        if add_action_output:
            next_first = first_rows[idx + 1] if idx + 1 < len(first_rows) else None
            left_delta, right_delta, left_bad, right_bad = _compute_delta_columns_for_file(
                left_curr=left_curr,
                right_curr=right_curr,
                left_gripper_curr=left_gripper_curr,
                right_gripper_curr=right_gripper_curr,
                ep_curr=ep_curr,
                next_first_row=next_first,
            )

            left_bad_total += left_bad
            right_bad_total += right_bad

            if left_dim is not None and left_dim < 0:
                left_dim = left_delta.shape[1]
                right_dim = right_delta.shape[1]
            elif left_dim != left_delta.shape[1] or right_dim != right_delta.shape[1]:
                raise ValueError(
                    f"Inconsistent output dim in {file_path}. "
                    f"left expected={left_dim}, got={left_delta.shape[1]}; "
                    f"right expected={right_dim}, got={right_delta.shape[1]}"
                )

            df[left_output_key] = [left_delta[i] for i in range(row_count)]
            df[right_output_key] = [right_delta[i] for i in range(row_count)]

        if add_state_output:
            left_state_rotvec, left_state_bad = _compute_state_rotvec_from_eef(left_curr)
            right_state_rotvec, right_state_bad = _compute_state_rotvec_from_eef(right_curr)
            left_state_bad_total += left_state_bad
            right_state_bad_total += right_state_bad

            if left_state_dim is not None and left_state_dim < 0:
                left_state_dim = left_state_rotvec.shape[1]
                right_state_dim = right_state_rotvec.shape[1]
            elif left_state_dim != left_state_rotvec.shape[1] or right_state_dim != right_state_rotvec.shape[1]:
                raise ValueError(
                    f"Inconsistent state rotvec dim in {file_path}. "
                    f"left expected={left_state_dim}, got={left_state_rotvec.shape[1]}; "
                    f"right expected={right_state_dim}, got={right_state_rotvec.shape[1]}"
                )

            df[left_state_rotvec_key] = [left_state_rotvec[i] for i in range(row_count)]
            df[right_state_rotvec_key] = [right_state_rotvec[i] for i in range(row_count)]

        if not dry_run:
            df.to_parquet(file_path, index=False)

    if add_action_output and ((left_dim is not None and left_dim < 0) or (right_dim is not None and right_dim < 0)):
        raise ValueError("No non-empty parquet files found to process for action deltas.")
    if add_state_output and (
        (left_state_dim is not None and left_state_dim < 0) or (right_state_dim is not None and right_state_dim < 0)
    ):
        raise ValueError("No non-empty parquet files found to process for state rotvec.")

    return (
        left_dim,
        right_dim,
        left_bad_total,
        right_bad_total,
        left_state_dim,
        right_state_dim,
        left_state_bad_total,
        right_state_bad_total,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add delta EEF action columns for RoboMIND Agilex 3RGB from obs_t -> obs_t+1 "
            "with per-episode boundary zero padding. Each output is [dx,dy,dz,rx,ry,rz,dg]. "
            "Also supports adding state orientation rotvec features for both arms."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Input dataset root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output dataset root. Required unless --in-place is used.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify dataset-root in place (no copy).",
    )
    parser.add_argument(
        "--left-output-key",
        type=str,
        default="actions.delta_eef_left",
        help="Output feature key for left arm delta EEF action.",
    )
    parser.add_argument(
        "--right-output-key",
        type=str,
        default="actions.delta_eef_right",
        help="Output feature key for right arm delta EEF action.",
    )
    parser.add_argument(
        "--left-state-rotvec-key",
        type=str,
        default=DEFAULT_LEFT_STATE_ROTVEC_KEY,
        help=f"Output state rotvec key for left arm (default: {DEFAULT_LEFT_STATE_ROTVEC_KEY}).",
    )
    parser.add_argument(
        "--right-state-rotvec-key",
        type=str,
        default=DEFAULT_RIGHT_STATE_ROTVEC_KEY,
        help=f"Output state rotvec key for right arm (default: {DEFAULT_RIGHT_STATE_ROTVEC_KEY}).",
    )
    parser.add_argument(
        "--state-rotvec-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only add state rotvec features/metadata and skip action delta outputs (default: false).",
    )
    parser.add_argument("--overwrite-output", action="store_true", help="Overwrite output-root if it exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate conversion on one sample file only.")
    parser.add_argument(
        "--recompute-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute stats for the new output keys and merge into existing stats (default: false).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    left_output_key = args.left_output_key
    right_output_key = args.right_output_key
    left_state_rotvec_key = args.left_state_rotvec_key
    right_state_rotvec_key = args.right_state_rotvec_key
    add_action_output = not args.state_rotvec_only
    add_state_output = True

    if left_output_key == right_output_key:
        raise ValueError("left-output-key and right-output-key must be different.")
    if left_state_rotvec_key == right_state_rotvec_key:
        raise ValueError("left-state-rotvec-key and right-state-rotvec-key must be different.")
    if left_state_rotvec_key in {left_output_key, right_output_key} or right_state_rotvec_key in {
        left_output_key,
        right_output_key,
    }:
        raise ValueError("State rotvec keys must differ from action output keys.")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    info_json = _load_json(info_path)
    existing_features = info_json.get("features", {})
    if add_action_output and (left_output_key in existing_features or right_output_key in existing_features):
        raise ValueError(
            f"Output keys already exist in info.json features. "
            f"left_exists={left_output_key in existing_features}, right_exists={right_output_key in existing_features}"
        )
    if left_state_rotvec_key in existing_features or right_state_rotvec_key in existing_features:
        raise ValueError(
            f"State rotvec keys already exist in info.json features. "
            f"left_exists={left_state_rotvec_key in existing_features}, right_exists={right_state_rotvec_key in existing_features}"
        )

    if args.dry_run:
        sample_file = _collect_data_files(dataset_root)[0]
        sample_df = pd.read_parquet(sample_file)
        if len(sample_df) == 0:
            raise ValueError(f"Sample parquet has no rows: {sample_file}")

        left_curr, right_curr, left_gripper_curr, right_gripper_curr, ep_curr = _extract_state_and_episode_arrays(
            sample_df.head(16)
        )
        left_state_rotvec, left_state_bad = _compute_state_rotvec_from_eef(left_curr)
        right_state_rotvec, right_state_bad = _compute_state_rotvec_from_eef(right_curr)
        left_delta = right_delta = None
        left_bad = right_bad = 0
        if add_action_output:
            left_delta, right_delta, left_bad, right_bad = _compute_delta_columns_for_file(
                left_curr=left_curr,
                right_curr=right_curr,
                left_gripper_curr=left_gripper_curr,
                right_gripper_curr=right_gripper_curr,
                ep_curr=ep_curr,
                next_first_row=None,
            )

        print("[DRY-RUN] Conversion validation succeeded on sample file:")
        print(f"  file={sample_file}")
        if add_action_output and left_delta is not None and right_delta is not None:
            print(f"  left_output_key={left_output_key}, shape={left_delta.shape}  # [dx,dy,dz,rx,ry,rz,dg]")
            print(f"  right_output_key={right_output_key}, shape={right_delta.shape}  # [dx,dy,dz,rx,ry,rz,dg]")
            print(f"  near_zero_quat_warnings_left_delta={left_bad}")
            print(f"  near_zero_quat_warnings_right_delta={right_bad}")
        print(
            f"  left_state_rotvec_key={left_state_rotvec_key}, shape={left_state_rotvec.shape}; "
            f"near_zero_quat_warnings_left_state={left_state_bad}"
        )
        print(
            f"  right_state_rotvec_key={right_state_rotvec_key}, shape={right_state_rotvec.shape}; "
            f"near_zero_quat_warnings_right_state={right_state_bad}"
        )
        return

    if args.in_place:
        output_root = dataset_root
        if args.output_root is not None:
            raise ValueError("Do not set --output-root together with --in-place.")
    else:
        if args.output_root is None:
            raise ValueError("--output-root is required unless --in-place is set.")
        output_root = args.output_root.resolve()
        if output_root == dataset_root:
            raise ValueError("output-root must be different from dataset-root unless --in-place is set.")

    _prepare_output_dir(
        dataset_root=dataset_root,
        output_root=output_root,
        overwrite_output=args.overwrite_output,
        in_place=args.in_place,
    )

    (
        left_dim,
        right_dim,
        left_bad_total,
        right_bad_total,
        left_state_dim,
        right_state_dim,
        left_state_bad_total,
        right_state_bad_total,
    ) = _process_data_files(
        output_root=output_root,
        left_output_key=left_output_key,
        right_output_key=right_output_key,
        left_state_rotvec_key=left_state_rotvec_key,
        right_state_rotvec_key=right_state_rotvec_key,
        dry_run=False,
        add_action_output=add_action_output,
        add_state_output=add_state_output,
    )

    _update_info_json(
        output_root,
        left_output_key,
        right_output_key,
        left_state_rotvec_key,
        right_state_rotvec_key,
        add_action_output=add_action_output,
        add_state_output=add_state_output,
    )
    _update_modality_json(
        output_root,
        left_output_key,
        right_output_key,
        left_state_rotvec_key,
        right_state_rotvec_key,
        add_action_output=add_action_output,
        add_state_output=add_state_output,
    )

    if args.recompute_stats:
        target_keys = {left_state_rotvec_key, right_state_rotvec_key}
        if add_action_output:
            target_keys |= {left_output_key, right_output_key}
        _recompute_stats(
            output_root,
            target_keys=target_keys,
            append_to_existing=True,
        )

    print("Completed RoboMIND Agilex delta EEF augmentation.")
    print(f"  output_root: {output_root}")
    print(f"  state_rotvec_only: {args.state_rotvec_only}")
    if add_action_output:
        print(f"  left key: {left_output_key} dim={left_dim}")
        print(f"  right key: {right_output_key} dim={right_dim}")
        print(f"  near-zero quaternion warnings (left delta): {left_bad_total}")
        print(f"  near-zero quaternion warnings (right delta): {right_bad_total}")
    print(f"  left state rotvec key: {left_state_rotvec_key} dim={left_state_dim}")
    print(f"  right state rotvec key: {right_state_rotvec_key} dim={right_state_dim}")
    print(f"  near-zero quaternion warnings (left state): {left_state_bad_total}")
    print(f"  near-zero quaternion warnings (right state): {right_state_bad_total}")
    print(f"  stats recomputed: {args.recompute_stats}")


if __name__ == "__main__":
    main()
