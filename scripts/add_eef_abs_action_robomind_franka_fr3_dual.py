#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm


FEATURE_NAMES = [
    "left_x",
    "left_y",
    "left_z",
    "left_rx",
    "left_ry",
    "left_rz",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_rx",
    "right_ry",
    "right_rz",
    "right_gripper",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def _collect_data_files(root: Path) -> list[Path]:
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")
    files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}/chunk-*/file-*.parquet")
    return files


def _prepare_output_dir(dataset_root: Path, output_root: Path, overwrite_output: bool) -> None:
    if output_root.exists():
        if not overwrite_output:
            raise FileExistsError(f"Output dir already exists: {output_root}. Use --overwrite-output to replace it.")
        shutil.rmtree(output_root)
    shutil.copytree(dataset_root, output_root)


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


def _stats_from_array(data: np.ndarray) -> dict[str, list[float]]:
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array for stats, got shape={data.shape}")
    if data.shape[0] == 0:
        raise ValueError("Cannot compute stats from empty array.")
    return {
        "min": data.min(axis=0).astype(np.float64).tolist(),
        "max": data.max(axis=0).astype(np.float64).tolist(),
        "mean": data.mean(axis=0).astype(np.float64).tolist(),
        "std": data.std(axis=0).astype(np.float64).tolist(),
        "count": [int(data.shape[0])],
        "q01": np.quantile(data, 0.01, axis=0).astype(np.float64).tolist(),
        "q10": np.quantile(data, 0.10, axis=0).astype(np.float64).tolist(),
        "q50": np.quantile(data, 0.50, axis=0).astype(np.float64).tolist(),
        "q90": np.quantile(data, 0.90, axis=0).astype(np.float64).tolist(),
        "q99": np.quantile(data, 0.99, axis=0).astype(np.float64).tolist(),
    }


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


class DualPandaForwardKinematics:
    def __init__(
        self,
        urdf_path: Path,
        left_ee_frame: str,
        right_ee_frame: str,
        left_base_frame: str,
        right_base_frame: str,
        kept_joint_names: list[str],
    ):
        self.urdf_path = urdf_path
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        try:
            import pinocchio as pin  # type: ignore
        except Exception as exc:
            raise ModuleNotFoundError(
                "Failed to import 'pinocchio'. Install Robot Pinocchio (not the unrelated homonymous package)."
            ) from exc

        if not hasattr(pin, "buildModelFromUrdf"):
            raise ModuleNotFoundError(
                "The imported 'pinocchio' module does not provide buildModelFromUrdf. "
                "This is likely not Robot Pinocchio. Install the robotics Pinocchio package."
            )

        self.pin = pin
        model = pin.buildModelFromUrdf(str(urdf_path))

        keep_joint_ids = []
        for name in kept_joint_names:
            jid = int(model.getJointId(name))
            if jid == 0:
                raise ValueError(f"Joint '{name}' not found in URDF: {urdf_path}")
            keep_joint_ids.append(jid)

        lock_joint_ids = [jid for jid in range(1, model.njoints) if jid not in keep_joint_ids]
        qref = np.zeros(model.nq, dtype=np.float64)
        self.model = pin.buildReducedModel(model, lock_joint_ids, qref)
        self.data = self.model.createData()
        self.nq = int(self.model.nq)

        if self.nq != len(kept_joint_names):
            raise ValueError(
                f"Reduced model nq mismatch: model.nq={self.nq}, expected={len(kept_joint_names)}. "
                f"URDF={urdf_path}"
            )

        frame_names = [self.model.frames[i].name for i in range(len(self.model.frames))]
        for frame in [left_ee_frame, right_ee_frame, left_base_frame, right_base_frame]:
            if not self.model.existFrame(frame):
                raise ValueError(
                    f"Frame '{frame}' not found in reduced URDF model {urdf_path}. Available frames: {frame_names}"
                )

        self.left_ee_id = self.model.getFrameId(left_ee_frame)
        self.right_ee_id = self.model.getFrameId(right_ee_frame)
        self.left_base_id = self.model.getFrameId(left_base_frame)
        self.right_base_id = self.model.getFrameId(right_base_frame)

    def compute_relative_poses(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = np.asarray(joints, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.nq:
            raise ValueError(f"Joint vector dim mismatch: got {q.shape[0]}, expected reduced nq={self.nq}")

        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)

        left_base = self.data.oMf[self.left_base_id]
        right_base = self.data.oMf[self.right_base_id]
        left_ee = self.data.oMf[self.left_ee_id]
        right_ee = self.data.oMf[self.right_ee_id]

        left_rel = left_base.inverse() * left_ee
        right_rel = right_base.inverse() * right_ee

        left_pos = np.asarray(left_rel.translation, dtype=np.float32).reshape(3)
        left_rot = np.asarray(left_rel.rotation, dtype=np.float32).reshape(3, 3)
        right_pos = np.asarray(right_rel.translation, dtype=np.float32).reshape(3)
        right_rot = np.asarray(right_rel.rotation, dtype=np.float32).reshape(3, 3)
        return left_pos, left_rot, right_pos, right_rot


@dataclass
class StateAlignmentResult:
    rows_checked: int
    pos_max_abs: float
    pos_mean_abs: float
    euler_max_abs: float
    euler_mean_abs: float


def _extract_dual_joint_rows(
    joints_arr: np.ndarray,
    left_joint_indices: list[int],
    right_joint_indices: list[int],
) -> np.ndarray:
    if joints_arr.ndim != 2:
        raise ValueError(f"Expected 2D joint array, got shape={joints_arr.shape}")
    max_idx = max(max(left_joint_indices), max(right_joint_indices))
    if joints_arr.shape[1] <= max_idx:
        raise ValueError(
            f"Joint dim mismatch: dim={joints_arr.shape[1]}, requires index {max_idx} for dual-arm extraction."
        )
    left = joints_arr[:, left_joint_indices]
    right = joints_arr[:, right_joint_indices]
    return np.concatenate([left, right], axis=1).astype(np.float32)


def _validate_state_fk_alignment(
    dataset_root: Path,
    fk: DualPandaForwardKinematics,
    obs_joint_key: str,
    obs_eef_key: str,
    left_joint_indices: list[int],
    right_joint_indices: list[int],
    obs_left_pos_indices: list[int],
    obs_left_euler_indices: list[int],
    obs_right_pos_indices: list[int],
    obs_right_euler_indices: list[int],
    euler_convention: str,
    validation_max_rows: int,
    validation_pos_tol: float,
    validation_euler_tol: float,
    show_progress: bool,
) -> StateAlignmentResult:
    files = _collect_data_files(dataset_root)
    pos_err_chunks: list[np.ndarray] = []
    euler_err_chunks: list[np.ndarray] = []

    remaining = validation_max_rows if validation_max_rows > 0 else None
    file_iter = files
    if show_progress:
        file_iter = tqdm(files, total=len(files), desc="[FK-ALIGNMENT] State check", unit="file")

    for file_path in file_iter:
        df = pd.read_parquet(file_path, columns=[obs_joint_key, obs_eef_key])
        if len(df) == 0:
            continue

        joints_arr = _to_2d_array(df[obs_joint_key], obs_joint_key)
        eef_arr = _to_2d_array(df[obs_eef_key], obs_eef_key)
        if joints_arr.shape[0] != eef_arr.shape[0]:
            raise ValueError(f"Row mismatch in {file_path}: joints={joints_arr.shape[0]} eef={eef_arr.shape[0]}")

        if remaining is not None:
            take = min(remaining, joints_arr.shape[0])
            joints_arr = joints_arr[:take]
            eef_arr = eef_arr[:take]
            remaining -= take

        max_eef_idx = max(
            max(obs_left_pos_indices),
            max(obs_left_euler_indices),
            max(obs_right_pos_indices),
            max(obs_right_euler_indices),
        )
        if eef_arr.shape[1] <= max_eef_idx:
            raise ValueError(
                f"Validation eef key '{obs_eef_key}' has dim={eef_arr.shape[1]} but indices need {max_eef_idx}."
            )

        q_rows = _extract_dual_joint_rows(
            joints_arr=joints_arr,
            left_joint_indices=left_joint_indices,
            right_joint_indices=right_joint_indices,
        )
        if q_rows.shape[1] != fk.nq:
            raise ValueError(f"Extracted dual joint dim={q_rows.shape[1]} but FK reduced model nq={fk.nq}.")

        pos_err = np.zeros((q_rows.shape[0], 6), dtype=np.float32)
        euler_err = np.zeros((q_rows.shape[0], 6), dtype=np.float32)

        row_iter = range(q_rows.shape[0])
        if show_progress:
            row_iter = tqdm(
                row_iter,
                total=q_rows.shape[0],
                desc=f"[FK-ALIGNMENT] {file_path.name}",
                unit="row",
                leave=False,
            )

        for i in row_iter:
            left_pos, left_rot, right_pos, right_rot = fk.compute_relative_poses(q_rows[i])
            fk_left_euler = R.from_matrix(left_rot).as_euler(euler_convention).astype(np.float32)
            fk_right_euler = R.from_matrix(right_rot).as_euler(euler_convention).astype(np.float32)

            obs_left_pos = eef_arr[i, obs_left_pos_indices].astype(np.float32)
            obs_left_euler = eef_arr[i, obs_left_euler_indices].astype(np.float32)
            obs_right_pos = eef_arr[i, obs_right_pos_indices].astype(np.float32)
            obs_right_euler = eef_arr[i, obs_right_euler_indices].astype(np.float32)

            pos_err[i, 0:3] = np.abs(left_pos - obs_left_pos)
            pos_err[i, 3:6] = np.abs(right_pos - obs_right_pos)
            euler_err[i, 0:3] = np.abs(_wrap_to_pi(fk_left_euler - obs_left_euler))
            euler_err[i, 3:6] = np.abs(_wrap_to_pi(fk_right_euler - obs_right_euler))

        pos_err_chunks.append(pos_err)
        euler_err_chunks.append(euler_err)

        if remaining is not None and remaining <= 0:
            break

    if not pos_err_chunks:
        raise ValueError("No rows found for state FK alignment validation.")

    pos_all = np.concatenate(pos_err_chunks, axis=0)
    euler_all = np.concatenate(euler_err_chunks, axis=0)
    result = StateAlignmentResult(
        rows_checked=int(pos_all.shape[0]),
        pos_max_abs=float(np.max(pos_all)),
        pos_mean_abs=float(np.mean(pos_all)),
        euler_max_abs=float(np.max(euler_all)),
        euler_mean_abs=float(np.mean(euler_all)),
    )

    print("[FK-ALIGNMENT] Validation summary")
    print(f"  rows_checked={result.rows_checked}")
    print(f"  euler_convention={euler_convention}")
    print(f"  pos_max_abs={result.pos_max_abs:.8f}, pos_mean_abs={result.pos_mean_abs:.8f}")
    print(f"  euler_max_abs={result.euler_max_abs:.8f}, euler_mean_abs={result.euler_mean_abs:.8f}")

    failed = []
    if result.pos_max_abs > validation_pos_tol:
        failed.append(f"position max abs {result.pos_max_abs:.8f} > tol {validation_pos_tol:.8f}")
    if result.euler_max_abs > validation_euler_tol:
        failed.append(f"euler max abs {result.euler_max_abs:.8f} > tol {validation_euler_tol:.8f}")
    if failed:
        raise ValueError(
            "State FK alignment validation failed. "
            "This usually indicates wrong URDF/frame/index mapping or Euler convention mismatch. "
            f"Details: {'; '.join(failed)}"
        )
    return result


def _compute_dual_eef_abs_for_file(
    df: pd.DataFrame,
    fk: DualPandaForwardKinematics,
    action_key: str,
    left_joint_indices: list[int],
    right_joint_indices: list[int],
    left_gripper_index: int,
    right_gripper_index: int,
    show_progress: bool,
    progress_desc: str | None = None,
) -> np.ndarray:
    if action_key not in df.columns:
        raise KeyError(f"Required action key '{action_key}' missing in parquet columns.")

    actions = _to_2d_array(df[action_key], action_key)
    if actions.shape[0] == 0:
        return np.zeros((0, 14), dtype=np.float32)

    max_required_idx = max(max(left_joint_indices), max(right_joint_indices), left_gripper_index, right_gripper_index)
    if actions.shape[1] <= max_required_idx:
        raise ValueError(
            f"Action dim mismatch for key '{action_key}': dim={actions.shape[1]}, requires index {max_required_idx}."
        )

    q_rows = _extract_dual_joint_rows(
        joints_arr=actions,
        left_joint_indices=left_joint_indices,
        right_joint_indices=right_joint_indices,
    )
    if q_rows.shape[1] != fk.nq:
        raise ValueError(f"Extracted dual joint dim={q_rows.shape[1]} but FK reduced model nq={fk.nq}.")

    out = np.zeros((actions.shape[0], 14), dtype=np.float32)
    row_iter = range(actions.shape[0])
    if show_progress and progress_desc is not None:
        row_iter = tqdm(row_iter, total=actions.shape[0], desc=progress_desc, unit="row", leave=False)

    for i in row_iter:
        left_pos, left_rot, right_pos, right_rot = fk.compute_relative_poses(q_rows[i])
        left_rotvec = R.from_matrix(left_rot).as_rotvec().astype(np.float32)
        right_rotvec = R.from_matrix(right_rot).as_rotvec().astype(np.float32)

        out[i, 0:3] = left_pos
        out[i, 3:6] = left_rotvec
        out[i, 6] = np.float32(actions[i, left_gripper_index])
        out[i, 7:10] = right_pos
        out[i, 10:13] = right_rotvec
        out[i, 13] = np.float32(actions[i, right_gripper_index])

    return out


def _update_info_json(info_path: Path, output_key: str) -> None:
    info = _load_json(info_path)
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Invalid info.json features in {info_path}")
    if output_key in features:
        raise ValueError(f"Output key '{output_key}' already exists in info.json features.")

    features[output_key] = {
        "dtype": "float32",
        "shape": [14],
        "names": {"motors": FEATURE_NAMES},
    }
    info["features"] = features
    _save_json(info_path, info)


def _update_modality_json(modality_path: Path, output_key: str) -> None:
    if not modality_path.exists():
        return

    modality = _load_json(modality_path)
    action_node = modality.get("action")
    if not isinstance(action_node, dict):
        raise ValueError(f"Invalid modality.json action node in {modality_path}")

    additions = {
        "left_eef_abs_position": {
            "start": 0,
            "end": 3,
            "absolute": True,
            "dtype": "float32",
            "original_key": output_key,
        },
        "left_eef_abs_orientation_rotvec": {
            "start": 3,
            "end": 6,
            "absolute": True,
            "dtype": "float32",
            "rotation_type": "axis_angle",
            "original_key": output_key,
        },
        "left_eef_abs_gripper": {
            "start": 6,
            "end": 7,
            "absolute": True,
            "dtype": "float32",
            "original_key": output_key,
        },
        "right_eef_abs_position": {
            "start": 7,
            "end": 10,
            "absolute": True,
            "dtype": "float32",
            "original_key": output_key,
        },
        "right_eef_abs_orientation_rotvec": {
            "start": 10,
            "end": 13,
            "absolute": True,
            "dtype": "float32",
            "rotation_type": "axis_angle",
            "original_key": output_key,
        },
        "right_eef_abs_gripper": {
            "start": 13,
            "end": 14,
            "absolute": True,
            "dtype": "float32",
            "original_key": output_key,
        },
    }

    duplicates = [k for k in additions if k in action_node]
    if duplicates:
        raise ValueError(f"modality.json action keys already exist: {duplicates}")
    action_node.update(additions)
    modality["action"] = action_node
    _save_json(modality_path, modality)


def _update_stats_json(stats_path: Path, output_key: str, stats_payload: dict[str, list[float]]) -> None:
    stats_data: dict[str, Any]
    if stats_path.exists():
        stats_data = _load_json(stats_path)
        if not isinstance(stats_data, dict):
            raise ValueError(f"Invalid stats json: {stats_path}")
    else:
        stats_data = {}

    if output_key in stats_data:
        raise ValueError(f"Output key '{output_key}' already exists in stats file: {stats_path}")
    stats_data[output_key] = stats_payload
    _save_json(stats_path, stats_data)


def _process_dataset(
    output_root: Path,
    fk: DualPandaForwardKinematics,
    action_key: str,
    left_joint_indices: list[int],
    right_joint_indices: list[int],
    left_gripper_index: int,
    right_gripper_index: int,
    output_key: str,
    recompute_stats: bool,
    show_progress: bool,
) -> tuple[int, int, dict[str, list[float]] | None]:
    files = _collect_data_files(output_root)
    all_chunks: list[np.ndarray] = []
    total_rows = 0
    total_files = 0

    file_iter = files
    if show_progress:
        file_iter = tqdm(files, total=len(files), desc="[CONVERT] Files", unit="file")

    for idx, file_path in enumerate(file_iter, start=1):
        df = pd.read_parquet(file_path)
        row_count = len(df)
        if row_count == 0:
            continue
        if output_key in df.columns:
            raise ValueError(f"Output key '{output_key}' already exists in parquet file: {file_path}")

        eef_abs = _compute_dual_eef_abs_for_file(
            df=df,
            fk=fk,
            action_key=action_key,
            left_joint_indices=left_joint_indices,
            right_joint_indices=right_joint_indices,
            left_gripper_index=left_gripper_index,
            right_gripper_index=right_gripper_index,
            show_progress=show_progress,
            progress_desc=f"[CONVERT] FK {file_path.name}",
        )

        total_rows += row_count
        total_files += 1
        if recompute_stats:
            all_chunks.append(eef_abs)

        df[output_key] = [eef_abs[i] for i in range(row_count)]
        df.to_parquet(file_path, index=False)

        if show_progress:
            file_iter.set_postfix_str(f"last={file_path.name} rows={row_count}")
        else:
            print(f"[{idx}/{len(files)}] processed {file_path} rows={row_count}")

    if total_rows == 0:
        raise ValueError("All parquet files are empty. Nothing was processed.")

    stats_payload = _stats_from_array(np.concatenate(all_chunks, axis=0)) if recompute_stats else None
    return total_files, total_rows, stats_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate absolute EEF actions for RoboMind dual-arm Franka dataset. "
            "Output layout is [l_xyz, l_rotvec, l_gripper, r_xyz, r_rotvec, r_gripper]."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/project_rlinf/jlchen/datasets/robomind_franka_fr3_dual"),
        help="Input dataset root directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output dataset root directory (default: <dataset-root>_with_eef_abs_action).",
    )
    parser.add_argument("--overwrite-output", action="store_true", help="Overwrite output directory if it exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate conversion settings without writing files.")
    parser.add_argument(
        "--run-conversion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Actually run dataset conversion and write files (default: false).",
    )
    parser.add_argument(
        "--recompute-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute stats entry for output key (default: true).",
    )
    parser.add_argument(
        "--precheck",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force state FK alignment precheck before real conversion (default: true).",
    )

    parser.add_argument(
        "--urdf-path",
        type=Path,
        default=Path("/mnt/project_rlinf/jlchen/code/lerobot/dual_panda.urdf"),
        help="Path to dual-arm Panda URDF used for FK.",
    )
    parser.add_argument("--left-ee-frame", type=str, default="panda_1_link8")
    parser.add_argument("--right-ee-frame", type=str, default="panda_2_link8")
    parser.add_argument("--left-base-frame", type=str, default="panda_1_link0")
    parser.add_argument("--right-base-frame", type=str, default="panda_2_link0")
    parser.add_argument(
        "--kept-joint-names",
        type=str,
        nargs="+",
        default=[
            "panda_1_joint1",
            "panda_1_joint2",
            "panda_1_joint3",
            "panda_1_joint4",
            "panda_1_joint5",
            "panda_1_joint6",
            "panda_1_joint7",
            "panda_2_joint1",
            "panda_2_joint2",
            "panda_2_joint3",
            "panda_2_joint4",
            "panda_2_joint5",
            "panda_2_joint6",
            "panda_2_joint7",
        ],
    )

    parser.add_argument("--action-key", type=str, default="actions.joint_position")
    parser.add_argument("--obs-joint-key", type=str, default="observation.states.joint_position")
    parser.add_argument("--obs-eef-key", type=str, default="observation.states.end_effector")
    parser.add_argument("--output-key", type=str, default="actions.eef_abs")

    parser.add_argument("--left-joint-indices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("--right-joint-indices", type=int, nargs="+", default=[8, 9, 10, 11, 12, 13, 14])
    parser.add_argument("--left-gripper-index", type=int, default=7)
    parser.add_argument("--right-gripper-index", type=int, default=15)

    parser.add_argument("--obs-left-pos-indices", type=int, nargs=3, default=[0, 1, 2])
    parser.add_argument("--obs-left-euler-indices", type=int, nargs=3, default=[3, 4, 5])
    parser.add_argument("--obs-right-pos-indices", type=int, nargs=3, default=[6, 7, 8])
    parser.add_argument("--obs-right-euler-indices", type=int, nargs=3, default=[9, 10, 11])
    parser.add_argument("--euler-convention", type=str, default="xyz")
    parser.add_argument(
        "--validate-state-fk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate state joint->eef consistency before conversion (default: true).",
    )
    parser.add_argument(
        "--validation-max-rows",
        type=int,
        default=0,
        help="Rows used for state alignment validation. 0 means all rows.",
    )
    parser.add_argument("--validation-pos-tol", type=float, default=1e-4)
    parser.add_argument("--validation-euler-tol", type=float, default=1e-4)
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars (default: true).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    if len(args.left_joint_indices) != 7 or len(args.right_joint_indices) != 7:
        raise ValueError("left/right joint indices must each contain exactly 7 indices for Panda arm joints.")
    if len(set(args.left_joint_indices)) != len(args.left_joint_indices):
        raise ValueError("--left-joint-indices must be unique.")
    if len(set(args.right_joint_indices)) != len(args.right_joint_indices):
        raise ValueError("--right-joint-indices must be unique.")
    if min(args.left_joint_indices + args.right_joint_indices) < 0:
        raise ValueError("Joint indices must be non-negative.")
    if args.left_gripper_index < 0 or args.right_gripper_index < 0:
        raise ValueError("Gripper indices must be non-negative.")

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    info_data = _load_json(info_path)
    features = info_data.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Invalid info.json features: {info_path}")
    for required_key in [args.action_key, args.obs_joint_key, args.obs_eef_key]:
        if required_key not in features:
            raise KeyError(f"Required key '{required_key}' not found in info.json features.")
    if args.output_key in features:
        raise ValueError(f"Output key '{args.output_key}' already exists in source info.json.")

    if args.output_root is None:
        output_root = dataset_root.parent / f"{dataset_root.name}_with_eef_abs_action"
    else:
        output_root = args.output_root.resolve()
    if args.run_conversion and not args.dry_run and output_root == dataset_root:
        raise ValueError("output-root must be different from dataset-root.")

    fk = DualPandaForwardKinematics(
        urdf_path=args.urdf_path.resolve(),
        left_ee_frame=args.left_ee_frame,
        right_ee_frame=args.right_ee_frame,
        left_base_frame=args.left_base_frame,
        right_base_frame=args.right_base_frame,
        kept_joint_names=args.kept_joint_names,
    )

    if fk.nq != len(args.kept_joint_names):
        raise RuntimeError("Internal FK setup error: reduced nq does not match kept-joint-names length.")

    should_precheck = args.validate_state_fk or (args.precheck and args.run_conversion and not args.dry_run)
    alignment_result = None
    if should_precheck:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alignment_result = _validate_state_fk_alignment(
                dataset_root=dataset_root,
                fk=fk,
                obs_joint_key=args.obs_joint_key,
                obs_eef_key=args.obs_eef_key,
                left_joint_indices=args.left_joint_indices,
                right_joint_indices=args.right_joint_indices,
                obs_left_pos_indices=args.obs_left_pos_indices,
                obs_left_euler_indices=args.obs_left_euler_indices,
                obs_right_pos_indices=args.obs_right_pos_indices,
                obs_right_euler_indices=args.obs_right_euler_indices,
                euler_convention=args.euler_convention,
                validation_max_rows=args.validation_max_rows,
                validation_pos_tol=args.validation_pos_tol,
                validation_euler_tol=args.validation_euler_tol,
                show_progress=args.show_progress,
            )

    if not args.run_conversion:
        print("Alignment-check-only mode finished. Conversion was skipped (--run-conversion is false).")
        print(f"  dataset_root={dataset_root}")
        print(f"  state_precheck_ran={should_precheck}")
        if alignment_result is not None:
            print(f"  rows_checked={alignment_result.rows_checked}")
        return

    if args.dry_run:
        data_files = _collect_data_files(dataset_root)
        sample_df = pd.read_parquet(data_files[0]).head(16)
        if len(sample_df) == 0:
            raise ValueError(f"Sample parquet has no rows: {data_files[0]}")
        sample_out = _compute_dual_eef_abs_for_file(
            df=sample_df,
            fk=fk,
            action_key=args.action_key,
            left_joint_indices=args.left_joint_indices,
            right_joint_indices=args.right_joint_indices,
            left_gripper_index=args.left_gripper_index,
            right_gripper_index=args.right_gripper_index,
            show_progress=args.show_progress,
            progress_desc="[DRY-RUN] FK sample",
        )
        print("[DRY-RUN] Validation succeeded.")
        print(f"  dataset_root={dataset_root}")
        print(f"  total_data_files={len(data_files)}")
        print(f"  sample_file={data_files[0]}")
        print(f"  output_key={args.output_key}")
        print(f"  sample_shape={sample_out.shape}")
        print(f"  sample_first_row={sample_out[0].tolist()}")
        if alignment_result is not None:
            print(f"  precheck_rows={alignment_result.rows_checked}")
            print(f"  precheck_pos_max_abs={alignment_result.pos_max_abs:.8f}")
            print(f"  precheck_euler_max_abs={alignment_result.euler_max_abs:.8f}")
        return

    _prepare_output_dir(dataset_root=dataset_root, output_root=output_root, overwrite_output=args.overwrite_output)

    total_files, total_rows, stats_payload = _process_dataset(
        output_root=output_root,
        fk=fk,
        action_key=args.action_key,
        left_joint_indices=args.left_joint_indices,
        right_joint_indices=args.right_joint_indices,
        left_gripper_index=args.left_gripper_index,
        right_gripper_index=args.right_gripper_index,
        output_key=args.output_key,
        recompute_stats=args.recompute_stats,
        show_progress=args.show_progress,
    )

    _update_info_json(output_root / "meta" / "info.json", args.output_key)
    _update_modality_json(output_root / "meta" / "modality.json", args.output_key)

    if args.recompute_stats:
        if stats_payload is None:
            raise ValueError("Internal error: stats payload is missing while recompute-stats is enabled.")
        stats_paths = [output_root / "meta" / "stats.json"]
        stats_gr00t_path = output_root / "meta" / "stats_gr00t.json"
        if stats_gr00t_path.exists():
            stats_paths.append(stats_gr00t_path)
        for sp in stats_paths:
            _update_stats_json(sp, args.output_key, stats_payload)

    print("Completed dual-arm EEF absolute-action augmentation.")
    print(f"  input_root: {dataset_root}")
    print(f"  output_root: {output_root}")
    print(f"  output_key: {args.output_key}")
    print(f"  feature_layout: {FEATURE_NAMES}")
    print(f"  files_processed: {total_files}")
    print(f"  rows_processed: {total_rows}")
    print(f"  stats_recomputed: {args.recompute_stats}")
    print(f"  state_precheck_ran: {should_precheck}")
    print(f"  run_conversion: {args.run_conversion}")
    if alignment_result is not None:
        print(f"  precheck_rows: {alignment_result.rows_checked}")
        print(f"  precheck_pos_max_abs: {alignment_result.pos_max_abs:.8f}")
        print(f"  precheck_euler_max_abs: {alignment_result.euler_max_abs:.8f}")


if __name__ == "__main__":
    main()
