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


ACTION_POS_KEY = "actions.end.position"
STATE_POS_KEY = "observation.states.end.position"
ACTION_QUAT_KEY = "actions.end.orientation"
STATE_QUAT_KEY = "observation.states.end.orientation"
ACTION_GRIP_KEY = "actions.effector.position"
STATE_GRIP_KEY = "observation.states.effector.position"
DEFAULT_STATE_ROTVEC_KEY = "observation.states.end.orientation_rotvec"


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


def _normalize_quat_xyzw(batch_quat: np.ndarray, key: str) -> np.ndarray:
    if batch_quat.ndim != 2 or batch_quat.shape[1] != 4:
        raise ValueError(f"{key} should have shape [N, 4], got {batch_quat.shape}.")
    norms = np.linalg.norm(batch_quat, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{key} contains near-zero-norm quaternion samples.")
    return batch_quat / norms


def _quat_delta_batch(q_action_xyzw: np.ndarray, q_state_xyzw: np.ndarray) -> np.ndarray:
    q_action_xyzw = _normalize_quat_xyzw(q_action_xyzw, "q_action_xyzw")
    q_state_xyzw = _normalize_quat_xyzw(q_state_xyzw, "q_state_xyzw")
    return (R.from_quat(q_action_xyzw) * R.from_quat(q_state_xyzw).inv()).as_quat().astype(np.float32)


def _compute_delta_action_matrix(df: pd.DataFrame) -> np.ndarray:
    for key in (
        ACTION_POS_KEY,
        STATE_POS_KEY,
        ACTION_QUAT_KEY,
        STATE_QUAT_KEY,
        ACTION_GRIP_KEY,
        STATE_GRIP_KEY,
    ):
        if key not in df.columns:
            raise KeyError(f"Required column missing in parquet: {key}")

    action_end_pos = _to_2d_array(df[ACTION_POS_KEY], ACTION_POS_KEY)
    state_end_pos = _to_2d_array(df[STATE_POS_KEY], STATE_POS_KEY)
    if action_end_pos.shape[1] != 6 or state_end_pos.shape[1] != 6:
        raise ValueError(
            f"Expected end position dim=6. action={action_end_pos.shape}, state={state_end_pos.shape}."
        )
    delta_end_pos = action_end_pos - state_end_pos

    action_quat = _to_2d_array(df[ACTION_QUAT_KEY], ACTION_QUAT_KEY)
    state_quat = _to_2d_array(df[STATE_QUAT_KEY], STATE_QUAT_KEY)
    if action_quat.shape[1] != 8 or state_quat.shape[1] != 8:
        raise ValueError(f"Expected end quaternion dim=8. action={action_quat.shape}, state={state_quat.shape}.")

    left_delta_quat_xyzw = _quat_delta_batch(action_quat[:, :4], state_quat[:, :4])
    right_delta_quat_xyzw = _quat_delta_batch(action_quat[:, 4:], state_quat[:, 4:])
    delta_end_ori_left = R.from_quat(left_delta_quat_xyzw).as_rotvec().astype(np.float32)
    delta_end_ori_right = R.from_quat(right_delta_quat_xyzw).as_rotvec().astype(np.float32)

    action_grip = _to_2d_array(df[ACTION_GRIP_KEY], ACTION_GRIP_KEY)
    state_grip = _to_2d_array(df[STATE_GRIP_KEY], STATE_GRIP_KEY)
    if action_grip.shape[1] != 2 or state_grip.shape[1] != 2:
        raise ValueError(f"Expected gripper dim=2. action={action_grip.shape}, state={state_grip.shape}.")
    delta_grip = action_grip - state_grip

    left_arm_delta = np.concatenate(
        [delta_end_pos[:, :3], delta_end_ori_left, delta_grip[:, :1]],
        axis=1,
    ).astype(np.float32)
    right_arm_delta = np.concatenate(
        [delta_end_pos[:, 3:], delta_end_ori_right, delta_grip[:, 1:]],
        axis=1,
    ).astype(np.float32)

    # Final layout: [left_arm(7), right_arm(7)]
    return np.concatenate([left_arm_delta, right_arm_delta], axis=1).astype(np.float32)


def _feature_names() -> list[str]:
    names = [
        "delta_end_pos_left_x",
        "delta_end_pos_left_y",
        "delta_end_pos_left_z",
        "delta_end_ori_left_rx",
        "delta_end_ori_left_ry",
        "delta_end_ori_left_rz",
        "delta_gripper_left",
        "delta_end_pos_right_x",
        "delta_end_pos_right_y",
        "delta_end_pos_right_z",
        "delta_end_ori_right_rx",
        "delta_end_ori_right_ry",
        "delta_end_ori_right_rz",
        "delta_gripper_right",
    ]
    return names


def _state_rotvec_feature_names() -> list[str]:
    return [
        "end_ori_left_rx",
        "end_ori_left_ry",
        "end_ori_left_rz",
        "end_ori_right_rx",
        "end_ori_right_ry",
        "end_ori_right_rz",
    ]


def _update_info_json(
    output_root: Path,
    output_key: str,
    state_rotvec_key: str,
    add_action_output: bool,
    add_state_output: bool,
) -> tuple[int | None, int | None]:
    info_path = output_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    info = _load_json(info_path)
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Invalid info.json features in {info_path}")
    if add_action_output and output_key in features:
        raise ValueError(f"Output key '{output_key}' already exists in info.json features.")
    if add_state_output and state_rotvec_key in features:
        raise ValueError(f"State rotvec key '{state_rotvec_key}' already exists in info.json features.")

    action_dim: int | None = None
    state_dim: int | None = None
    if add_action_output:
        action_names = _feature_names()
        features[output_key] = {"dtype": "float32", "shape": [len(action_names)], "names": {"motors": action_names}}
        action_dim = len(action_names)
    if add_state_output:
        state_names = _state_rotvec_feature_names()
        features[state_rotvec_key] = {"dtype": "float32", "shape": [len(state_names)], "names": {"motors": state_names}}
        state_dim = len(state_names)
    info["features"] = features
    _save_json(info_path, info)
    return action_dim, state_dim


def _update_modality_json(
    output_root: Path,
    output_key: str,
    state_rotvec_key: str,
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
            "delta_end_position_left",
            "delta_end_orientation_left",
            "delta_gripper_left",
            "delta_end_position_right",
            "delta_end_orientation_right",
            "delta_gripper_right",
        ]
        duplicate_keys = [k for k in new_keys if k in action_node]
        if duplicate_keys:
            raise ValueError(f"modality.json action keys already exist: {duplicate_keys}")

        cursor = 0
        action_node["delta_end_position_left"] = {
            "start": cursor,
            "end": cursor + 3,
            "original_key": output_key,
            "absolute": False,
            "dtype": "float32",
        }
        cursor += 3
        action_node["delta_end_orientation_left"] = {
            "start": cursor,
            "end": cursor + 3,
            "rotation_type": "axis_angle",
            "original_key": output_key,
            "absolute": False,
            "dtype": "float32",
        }
        cursor += 3
        action_node["delta_gripper_left"] = {
            "start": cursor,
            "end": cursor + 1,
            "original_key": output_key,
            "absolute": False,
            "dtype": "float32",
        }
        cursor += 1
        action_node["delta_end_position_right"] = {
            "start": cursor,
            "end": cursor + 3,
            "original_key": output_key,
            "absolute": False,
            "dtype": "float32",
        }
        cursor += 3
        action_node["delta_end_orientation_right"] = {
            "start": cursor,
            "end": cursor + 3,
            "rotation_type": "axis_angle",
            "original_key": output_key,
            "absolute": False,
            "dtype": "float32",
        }
        cursor += 3
        action_node["delta_gripper_right"] = {
            "start": cursor,
            "end": cursor + 1,
            "original_key": output_key,
            "absolute": False,
            "dtype": "float32",
        }

    if add_state_output:
        if "state" not in modality or not isinstance(modality["state"], dict):
            raise ValueError("modality.json must contain top-level 'state' dict.")
        state_node = modality["state"]
        state_new_keys = [
            "end_orientation_left_rotvec",
            "end_orientation_right_rotvec",
        ]
        state_duplicates = [k for k in state_new_keys if k in state_node]
        if state_duplicates:
            raise ValueError(f"modality.json state keys already exist: {state_duplicates}")

        state_node["end_orientation_left_rotvec"] = {
            "start": 0,
            "end": 3,
            "rotation_type": "axis_angle",
            "original_key": state_rotvec_key,
            "absolute": True,
            "dtype": "float32",
        }
        state_node["end_orientation_right_rotvec"] = {
            "start": 3,
            "end": 6,
            "rotation_type": "axis_angle",
            "original_key": state_rotvec_key,
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


def _compute_state_rotvec_matrix(df: pd.DataFrame) -> np.ndarray:
    if STATE_QUAT_KEY not in df.columns:
        raise KeyError(f"Required column missing in parquet: {STATE_QUAT_KEY}")
    state_quat = _to_2d_array(df[STATE_QUAT_KEY], STATE_QUAT_KEY)
    if state_quat.shape[1] != 8:
        raise ValueError(f"Expected state quaternion dim=8, got {state_quat.shape}.")
    state_left_rotvec = R.from_quat(_normalize_quat_xyzw(state_quat[:, :4], f"{STATE_QUAT_KEY}.left")).as_rotvec().astype(np.float32)
    state_right_rotvec = R.from_quat(_normalize_quat_xyzw(state_quat[:, 4:], f"{STATE_QUAT_KEY}.right")).as_rotvec().astype(np.float32)
    return np.concatenate([state_left_rotvec, state_right_rotvec], axis=1).astype(np.float32)


def _process_data_files(
    output_root: Path,
    output_key: str,
    state_rotvec_key: str,
    dry_run: bool,
    add_action_output: bool,
    add_state_output: bool,
) -> tuple[int | None, int | None]:
    files = _collect_data_files(output_root)
    output_dim = -1 if add_action_output else None
    state_rotvec_dim = -1 if add_state_output else None

    for file_path in files:
        df = pd.read_parquet(file_path)
        row_count = len(df)
        if row_count == 0:
            continue
        if add_action_output and output_key in df.columns:
            raise ValueError(f"Output key '{output_key}' already exists in file {file_path}.")
        if add_state_output and state_rotvec_key in df.columns:
            raise ValueError(f"State rotvec key '{state_rotvec_key}' already exists in file {file_path}.")

        if add_action_output:
            delta_action = _compute_delta_action_matrix(df)
        else:
            delta_action = None
        if add_state_output:
            state_rotvec = _compute_state_rotvec_matrix(df)
        else:
            state_rotvec = None

        if add_action_output and delta_action is not None:
            if output_dim is not None and output_dim < 0:
                output_dim = delta_action.shape[1]
            elif output_dim != delta_action.shape[1]:
                raise ValueError(
                    f"Inconsistent output dim across files. expected {output_dim}, got {delta_action.shape[1]} "
                    f"for {file_path}"
                )
        if add_state_output and state_rotvec is not None:
            if state_rotvec_dim is not None and state_rotvec_dim < 0:
                state_rotvec_dim = state_rotvec.shape[1]
            elif state_rotvec_dim != state_rotvec.shape[1]:
                raise ValueError(
                    f"Inconsistent state rotvec dim across files. expected {state_rotvec_dim}, got {state_rotvec.shape[1]} "
                    f"for {file_path}"
                )

        if add_action_output and delta_action is not None:
            df[output_key] = [delta_action[i] for i in range(row_count)]
        if add_state_output and state_rotvec is not None:
            df[state_rotvec_key] = [state_rotvec[i] for i in range(row_count)]
        if not dry_run:
            df.to_parquet(file_path, index=False)

    if add_action_output and output_dim is not None and output_dim < 0:
        raise ValueError("No non-empty parquet files found to process for action output.")
    if add_state_output and state_rotvec_dim is not None and state_rotvec_dim < 0:
        raise ValueError("No non-empty parquet files found to process for state rotvec output.")
    return output_dim, state_rotvec_dim


def _prepare_output_dir(dataset_root: Path, output_root: Path, overwrite_output: bool, in_place: bool) -> None:
    if in_place:
        return
    if output_root.exists():
        if not overwrite_output:
            raise FileExistsError(f"Output dir already exists: {output_root}. Use --overwrite-output to replace it.")
        shutil.rmtree(output_root)
    shutil.copytree(dataset_root, output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute AgiBot delta actions with fixed layout [left(7), right(7)] where each arm is "
            "[delta_position(3), delta_orientation_axis_angle(3), delta_gripper(1)], "
            "write a new action vector key and state rotvec key, update info/modality metadata, "
            "and optionally recompute stats."
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
        "--output-key",
        type=str,
        default="actions.delta",
        help="New feature key written into parquet/meta info (default: actions.delta).",
    )
    parser.add_argument(
        "--state-rotvec-key",
        type=str,
        default=DEFAULT_STATE_ROTVEC_KEY,
        help=(
            "New state rotvec feature key converted from observation.states.end.orientation "
            f"(default: {DEFAULT_STATE_ROTVEC_KEY})."
        ),
    )
    parser.add_argument(
        "--state-rotvec-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Only add state rotvec key/metadata and skip action delta generation. "
            "Useful when actions.delta already exists."
        ),
    )
    parser.add_argument("--overwrite-output", action="store_true", help="Overwrite output-root if it exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate conversion on one sample file only.")
    parser.add_argument(
        "--recompute-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Recompute stats for output-key and merge into existing stats (default: false).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_key = args.output_key
    state_rotvec_key = args.state_rotvec_key
    add_action_output = not args.state_rotvec_only
    add_state_output = True

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if output_key == state_rotvec_key:
        raise ValueError("--state-rotvec-key must differ from --output-key.")

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    info_json = _load_json(info_path)
    if add_action_output and output_key in info_json.get("features", {}):
        raise ValueError(f"Output key '{output_key}' already exists in info.json features.")
    if add_state_output and state_rotvec_key in info_json.get("features", {}):
        raise ValueError(f"State rotvec key '{state_rotvec_key}' already exists in info.json features.")

    if args.dry_run:
        sample_file = _collect_data_files(dataset_root)[0]
        sample_df = pd.read_parquet(sample_file)
        if len(sample_df) == 0:
            raise ValueError(f"Sample parquet has no rows: {sample_file}")
        sample_df = sample_df.head(8)
        sample_delta = _compute_delta_action_matrix(sample_df) if add_action_output else None
        sample_state_rotvec = _compute_state_rotvec_matrix(sample_df) if add_state_output else None
        print("[DRY-RUN] Conversion validation succeeded on sample file:")
        print(f"  file={sample_file}")
        if add_action_output:
            print(f"  output_key={output_key}")
        print(f"  state_rotvec_key={state_rotvec_key}")
        print("  layout=[left(7), right(7)]")
        if sample_delta is not None:
            print(f"  action_delta_shape={sample_delta.shape}")
        if sample_state_rotvec is not None:
            print(f"  state_rotvec_shape={sample_state_rotvec.shape}")
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
    output_dim, state_rotvec_dim = _process_data_files(
        output_root,
        output_key,
        state_rotvec_key,
        dry_run=False,
        add_action_output=add_action_output,
        add_state_output=add_state_output,
    )
    info_action_dim, info_state_dim = _update_info_json(
        output_root,
        output_key,
        state_rotvec_key,
        add_action_output=add_action_output,
        add_state_output=add_state_output,
    )
    if add_action_output and output_dim != info_action_dim:
        raise ValueError(f"Computed action output_dim={output_dim} does not match info_dim={info_action_dim}.")
    if add_state_output and state_rotvec_dim != info_state_dim:
        raise ValueError(
            f"Computed state_rotvec_dim={state_rotvec_dim} does not match info_dim={info_state_dim}."
        )
    _update_modality_json(
        output_root,
        output_key,
        state_rotvec_key,
        add_action_output=add_action_output,
        add_state_output=add_state_output,
    )

    if args.recompute_stats:
        target_keys = {state_rotvec_key}
        if add_action_output:
            target_keys.add(output_key)
        _recompute_stats(output_root, target_keys=target_keys, append_to_existing=True)

    print("Completed AgiBot delta-action augmentation.")
    print(f"  output_root: {output_root}")
    print(f"  state_rotvec_only: {args.state_rotvec_only}")
    if add_action_output:
        print(f"  output_key: {output_key}")
    print(f"  state_rotvec_key: {state_rotvec_key}")
    print("  layout: [left(7), right(7)]")
    if add_action_output:
        print(f"  action dim: {output_dim}")
    print(f"  state rotvec dim: {state_rotvec_dim}")
    print(f"  stats recomputed: {args.recompute_stats}")


if __name__ == "__main__":
    main()
