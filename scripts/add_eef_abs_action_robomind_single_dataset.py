#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import warnings
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm


FEATURE_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]


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


@dataclass
class AlignmentResult:
    euler_convention: str
    pos_max_abs: float
    pos_mean_abs: float
    rot_angle_max_abs: float
    rot_angle_mean_abs: float
    euler_max_abs: float
    euler_mean_abs: float


class ForwardKinematics:
    def __init__(
        self,
        urdf_path: Path,
        ee_frame_name: str,
        tcp_offset_xyz: np.ndarray,
        tcp_offset_rpy: np.ndarray,
        tcp_offset_euler_convention: str,
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
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        if not self.model.existFrame(ee_frame_name):
            available_frames = [self.model.frames[i].name for i in range(len(self.model.frames))]
            raise ValueError(
                f"Frame '{ee_frame_name}' not found in URDF {urdf_path}. Available frames: {available_frames}"
            )

        self.ee_frame_name = ee_frame_name
        self.ee_id = self.model.getFrameId(ee_frame_name)
        self.nq = int(self.model.nq)

        self.tcp_offset_xyz = np.asarray(tcp_offset_xyz, dtype=np.float64).reshape(3)
        self.tcp_offset_rotmat = R.from_euler(
            tcp_offset_euler_convention,
            np.asarray(tcp_offset_rpy, dtype=np.float64).reshape(3),
        ).as_matrix()

    def compute_pose(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(joints, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.nq:
            raise ValueError(
                f"Joint vector dim mismatch: got {q.shape[0]}, model.nq={self.nq}. "
                f"URDF={self.urdf_path}, frame={self.ee_frame_name}"
            )

        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.ee_id]

        ee_pos = np.asarray(pose.translation, dtype=np.float64).reshape(3)
        ee_rot = np.asarray(pose.rotation, dtype=np.float64).reshape(3, 3)

        # world_T_tcp = world_T_ee * ee_T_tcp
        tcp_pos = ee_pos + ee_rot @ self.tcp_offset_xyz
        tcp_rot = ee_rot @ self.tcp_offset_rotmat

        return tcp_pos.astype(np.float32), tcp_rot.astype(np.float32)


def _compute_eef_abs_for_file(
    df: pd.DataFrame,
    fk: ForwardKinematics,
    action_key: str,
    joint_indices: list[int],
    gripper_action_index: int,
    show_progress: bool,
    progress_desc: str | None = None,
) -> np.ndarray:
    if action_key not in df.columns:
        raise KeyError(f"Required action key '{action_key}' missing in parquet columns.")

    actions = _to_2d_array(df[action_key], action_key)
    if actions.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float32)

    max_required_index = max(max(joint_indices), gripper_action_index)
    if actions.shape[1] <= max_required_index:
        raise ValueError(
            f"Action dim mismatch for key '{action_key}': dim={actions.shape[1]}, "
            f"requires index {max_required_index}."
        )

    joints = actions[:, joint_indices]
    gripper = actions[:, gripper_action_index]

    out = np.zeros((actions.shape[0], 7), dtype=np.float32)
    row_iter = range(actions.shape[0])
    if show_progress and progress_desc is not None:
        row_iter = tqdm(row_iter, total=actions.shape[0], desc=progress_desc, unit="row", leave=False)

    for i in row_iter:
        pos, rotmat = fk.compute_pose(joints[i])
        out[i, :3] = pos
        out[i, 3:6] = R.from_matrix(rotmat).as_rotvec().astype(np.float32)
    out[:, 6] = gripper.astype(np.float32)
    return out


def _collect_validation_arrays(
    dataset_root: Path,
    obs_joint_key: str,
    obs_eef_key: str,
    validation_max_rows: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray]:
    files = _collect_data_files(dataset_root)
    joints_chunks: list[np.ndarray] = []
    eef_chunks: list[np.ndarray] = []
    remaining = validation_max_rows

    file_iter = files
    if show_progress:
        file_iter = tqdm(files, total=len(files), desc="[FK-ALIGNMENT] Load parquet", unit="file")

    for file_path in file_iter:
        df = pd.read_parquet(file_path, columns=[obs_joint_key, obs_eef_key])
        if len(df) == 0:
            continue

        joints_arr = _to_2d_array(df[obs_joint_key], obs_joint_key)
        eef_arr = _to_2d_array(df[obs_eef_key], obs_eef_key)
        if joints_arr.shape[0] != eef_arr.shape[0]:
            raise ValueError(
                f"Row mismatch in {file_path}: joints={joints_arr.shape[0]} eef={eef_arr.shape[0]}"
            )

        if validation_max_rows > 0:
            take = min(remaining, joints_arr.shape[0])
            joints_arr = joints_arr[:take]
            eef_arr = eef_arr[:take]
            remaining -= take

        joints_chunks.append(joints_arr)
        eef_chunks.append(eef_arr)

        if validation_max_rows > 0 and remaining <= 0:
            break

    if not joints_chunks:
        raise ValueError("No rows found for FK alignment validation.")

    joints = np.concatenate(joints_chunks, axis=0)
    eef = np.concatenate(eef_chunks, axis=0)
    return joints, eef


def _euler_candidates() -> list[str]:
    seqs = ["".join(p) for p in permutations("xyz", 3)]
    return seqs + [s.upper() for s in seqs]


def _evaluate_alignment(
    fk_rotations: np.ndarray,
    obs_euler: np.ndarray,
    seq: str,
) -> tuple[float, float, float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs_rot = R.from_euler(seq, obs_euler)
        fk_rot = R.from_matrix(fk_rotations)
        delta = fk_rot * obs_rot.inv()
        rot_angle_abs = np.abs(delta.magnitude())

        fk_euler = fk_rot.as_euler(seq)
        euler_diff = np.abs(_wrap_to_pi(fk_euler - obs_euler))

    return (
        float(np.max(rot_angle_abs)),
        float(np.mean(rot_angle_abs)),
        float(np.max(euler_diff)),
        float(np.mean(euler_diff)),
    )


def _validate_fk_alignment(
    dataset_root: Path,
    fk: ForwardKinematics,
    obs_joint_key: str,
    obs_eef_key: str,
    joint_indices: list[int],
    obs_eef_pos_indices: list[int],
    obs_eef_euler_indices: list[int],
    euler_convention: str,
    validation_max_rows: int,
    validation_pos_tol: float,
    validation_rot_angle_tol: float,
    validation_euler_tol: float,
    show_progress: bool,
) -> AlignmentResult:
    joints_all, eef_all = _collect_validation_arrays(
        dataset_root=dataset_root,
        obs_joint_key=obs_joint_key,
        obs_eef_key=obs_eef_key,
        validation_max_rows=validation_max_rows,
        show_progress=show_progress,
    )

    max_joint_index = max(joint_indices)
    if joints_all.shape[1] <= max_joint_index:
        raise ValueError(
            f"Validation joint key '{obs_joint_key}' has dim={joints_all.shape[1]} but needs index {max_joint_index}."
        )

    if eef_all.shape[1] <= max(max(obs_eef_pos_indices), max(obs_eef_euler_indices)):
        raise ValueError(
            f"Validation eef key '{obs_eef_key}' has dim={eef_all.shape[1]} but indices are out of range."
        )

    joints = joints_all[:, joint_indices]
    obs_pos = eef_all[:, obs_eef_pos_indices].astype(np.float32)
    obs_euler = eef_all[:, obs_eef_euler_indices].astype(np.float32)

    fk_pos = np.zeros_like(obs_pos, dtype=np.float32)
    fk_rotmats = np.zeros((obs_pos.shape[0], 3, 3), dtype=np.float32)

    row_iter = range(obs_pos.shape[0])
    if show_progress:
        row_iter = tqdm(row_iter, total=obs_pos.shape[0], desc="[FK-ALIGNMENT] FK", unit="row")

    for i in row_iter:
        pos, rotmat = fk.compute_pose(joints[i])
        fk_pos[i] = pos
        fk_rotmats[i] = rotmat

    pos_diff = np.abs(fk_pos - obs_pos)
    pos_max_abs = float(np.max(pos_diff))
    pos_mean_abs = float(np.mean(pos_diff))

    if euler_convention.lower() == "auto":
        candidates = _euler_candidates()
    else:
        candidates = [euler_convention]

    best_seq = None
    best_tuple = None
    for seq in candidates:
        try:
            metrics = _evaluate_alignment(fk_rotations=fk_rotmats, obs_euler=obs_euler, seq=seq)
        except Exception:
            continue
        ranking = (metrics[1], metrics[3], metrics[0], metrics[2])
        if best_tuple is None or ranking < best_tuple:
            best_tuple = ranking
            best_seq = (seq, metrics)

    if best_seq is None:
        raise ValueError(
            "Failed to evaluate any Euler convention for alignment. "
            "Please check observation EEF orientation encoding."
        )

    chosen_seq, (rot_angle_max_abs, rot_angle_mean_abs, euler_max_abs, euler_mean_abs) = best_seq

    result = AlignmentResult(
        euler_convention=chosen_seq,
        pos_max_abs=pos_max_abs,
        pos_mean_abs=pos_mean_abs,
        rot_angle_max_abs=rot_angle_max_abs,
        rot_angle_mean_abs=rot_angle_mean_abs,
        euler_max_abs=euler_max_abs,
        euler_mean_abs=euler_mean_abs,
    )

    print("[FK-ALIGNMENT] Validation summary")
    print(f"  rows_checked={obs_pos.shape[0]}")
    print(f"  chosen_euler_convention={result.euler_convention}")
    print(f"  pos_max_abs={result.pos_max_abs:.8f}, pos_mean_abs={result.pos_mean_abs:.8f}")
    print(
        f"  rot_angle_max_abs={result.rot_angle_max_abs:.8f}, "
        f"rot_angle_mean_abs={result.rot_angle_mean_abs:.8f}"
    )
    print(
        f"  euler_max_abs={result.euler_max_abs:.8f}, "
        f"euler_mean_abs={result.euler_mean_abs:.8f}"
    )

    failed = []
    if result.pos_max_abs > validation_pos_tol:
        failed.append(
            f"position max abs {result.pos_max_abs:.8f} > tol {validation_pos_tol:.8f}"
        )
    if result.rot_angle_max_abs > validation_rot_angle_tol:
        failed.append(
            f"rotation angle max abs {result.rot_angle_max_abs:.8f} > tol {validation_rot_angle_tol:.8f}"
        )
    if result.euler_max_abs > validation_euler_tol:
        failed.append(
            f"euler component max abs {result.euler_max_abs:.8f} > tol {validation_euler_tol:.8f}"
        )

    if failed:
        joined = "; ".join(failed)
        raise ValueError(
            "FK alignment validation failed. "
            "This usually indicates wrong URDF/EEF frame/TCP offset or Euler convention mismatch. "
            f"Details: {joined}"
        )

    return result


def _update_info_json(info_path: Path, output_key: str) -> None:
    info = _load_json(info_path)
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Invalid info.json features in {info_path}")
    if output_key in features:
        raise ValueError(f"Output key '{output_key}' already exists in info.json features.")

    features[output_key] = {
        "dtype": "float32",
        "shape": [7],
        "names": {"motors": FEATURE_NAMES},
    }
    info["features"] = features
    _save_json(info_path, info)


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
    fk: ForwardKinematics,
    action_key: str,
    joint_indices: list[int],
    gripper_action_index: int,
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

        eef_abs = _compute_eef_abs_for_file(
            df=df,
            fk=fk,
            action_key=action_key,
            joint_indices=joint_indices,
            gripper_action_index=gripper_action_index,
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
            "Generate absolute EEF actions for RoboMind single-arm Franka datasets. "
            "Output layout is [x, y, z, rx, ry, rz, gripper] with rotation in rotvec."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Input dataset root directory.")
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
        help="Recompute stats.json entry for output key (default: true).",
    )

    parser.add_argument(
        "--urdf-path",
        type=Path,
        default=Path("/mnt/project_rlinf/jlchen/code/franka_ros/panda.urdf"),
        help="Path to Franka URDF used for FK.",
    )
    parser.add_argument(
        "--ee-frame-name",
        type=str,
        default="panda_link8",
        help="End-effector frame name in URDF.",
    )
    parser.add_argument(
        "--tcp-offset-xyz",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Optional TCP offset translation in EE frame, meters.",
    )
    parser.add_argument(
        "--tcp-offset-rpy",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Optional TCP offset Euler angles in EE frame, radians.",
    )
    parser.add_argument(
        "--tcp-offset-euler-convention",
        type=str,
        default="xyz",
        help="Euler convention for --tcp-offset-rpy.",
    )

    parser.add_argument(
        "--action-key",
        type=str,
        default="actions.joint_position",
        help="Action key containing joint targets.",
    )
    parser.add_argument(
        "--joint-indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6],
        help="Indices for arm joints in action vector.",
    )
    parser.add_argument(
        "--gripper-action-index",
        type=int,
        default=7,
        help="Index for gripper action in action vector.",
    )
    parser.add_argument(
        "--output-key",
        type=str,
        default="actions.eef_abs",
        help="Output key name for absolute EEF action.",
    )

    parser.add_argument(
        "--validate-fk-alignment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate FK frame alignment against observation before conversion (default: true).",
    )
    parser.add_argument(
        "--obs-joint-key",
        type=str,
        default="observation.states.joint_position",
        help="Observation joint key used in FK alignment check.",
    )
    parser.add_argument(
        "--obs-eef-key",
        type=str,
        default="observation.states.end_effector",
        help="Observation EEF key used in FK alignment check.",
    )
    parser.add_argument(
        "--obs-eef-pos-indices",
        type=int,
        nargs=3,
        default=[0, 1, 2],
        help="Position indices inside --obs-eef-key.",
    )
    parser.add_argument(
        "--obs-eef-euler-indices",
        type=int,
        nargs=3,
        default=[3, 4, 5],
        help="Euler indices inside --obs-eef-key.",
    )
    parser.add_argument(
        "--euler-convention",
        type=str,
        default="auto",
        help="Euler convention for obs EEF orientation check; use 'auto' to search.",
    )
    parser.add_argument(
        "--validation-max-rows",
        type=int,
        default=100,
        help="Rows used for alignment validation. 0 means all rows (default).",
    )
    parser.add_argument(
        "--validation-pos-tol",
        type=float,
        default=1e-4,
        help="Max abs position error tolerance for alignment validation.",
    )
    parser.add_argument(
        "--validation-rot-angle-tol",
        type=float,
        default=1e-4,
        help="Max abs rotation-angle error tolerance (radians).",
    )
    parser.add_argument(
        "--validation-euler-tol",
        type=float,
        default=1e-4,
        help="Max abs Euler component error tolerance (radians).",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars for validation and conversion (default: true).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    if len(args.joint_indices) == 0:
        raise ValueError("--joint-indices must not be empty.")
    if len(set(args.joint_indices)) != len(args.joint_indices):
        raise ValueError("--joint-indices must be unique.")
    if min(args.joint_indices) < 0:
        raise ValueError("--joint-indices must be non-negative.")
    if args.gripper_action_index < 0:
        raise ValueError("--gripper-action-index must be non-negative.")

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    info_data = _load_json(info_path)
    features = info_data.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Invalid info.json features: {info_path}")
    if args.action_key not in features:
        raise KeyError(f"Action key '{args.action_key}' not found in info.json features.")
    if args.output_key in features:
        raise ValueError(f"Output key '{args.output_key}' already exists in source info.json.")

    if args.output_root is None:
        output_root = dataset_root.parent / f"{dataset_root.name}_with_eef_abs_action"
    else:
        output_root = args.output_root.resolve()

    if args.run_conversion and not args.dry_run and output_root == dataset_root:
        raise ValueError("output-root must be different from dataset-root.")

    fk: ForwardKinematics | None = None
    if args.validate_fk_alignment or args.run_conversion:
        fk = ForwardKinematics(
            urdf_path=args.urdf_path.resolve(),
            ee_frame_name=args.ee_frame_name,
            tcp_offset_xyz=np.asarray(args.tcp_offset_xyz, dtype=np.float64),
            tcp_offset_rpy=np.asarray(args.tcp_offset_rpy, dtype=np.float64),
            tcp_offset_euler_convention=args.tcp_offset_euler_convention,
        )

        if len(args.joint_indices) != fk.nq:
            raise ValueError(
                f"Joint count mismatch: len(--joint-indices)={len(args.joint_indices)} but FK model.nq={fk.nq}."
            )

    alignment_result = None
    if args.validate_fk_alignment:
        if fk is None:
            raise RuntimeError("Internal error: FK engine is not initialized for alignment validation.")
        alignment_result = _validate_fk_alignment(
            dataset_root=dataset_root,
            fk=fk,
            obs_joint_key=args.obs_joint_key,
            obs_eef_key=args.obs_eef_key,
            joint_indices=args.joint_indices,
            obs_eef_pos_indices=args.obs_eef_pos_indices,
            obs_eef_euler_indices=args.obs_eef_euler_indices,
            euler_convention=args.euler_convention,
            validation_max_rows=args.validation_max_rows,
            validation_pos_tol=args.validation_pos_tol,
            validation_rot_angle_tol=args.validation_rot_angle_tol,
            validation_euler_tol=args.validation_euler_tol,
            show_progress=args.show_progress,
        )

    if not args.run_conversion:
        print("Alignment-check-only mode finished. Conversion was skipped (--run-conversion is false).")
        print(f"  dataset_root={dataset_root}")
        print(f"  validate_fk_alignment={args.validate_fk_alignment}")
        if alignment_result is not None:
            print(f"  fk_alignment_convention={alignment_result.euler_convention}")
        return

    if args.dry_run:
        if fk is None:
            raise RuntimeError("Internal error: FK engine is not initialized for dry-run conversion validation.")
        data_files = _collect_data_files(dataset_root)
        sample_df = pd.read_parquet(data_files[0]).head(16)
        if len(sample_df) == 0:
            raise ValueError(f"Sample parquet has no rows: {data_files[0]}")

        sample_eef_abs = _compute_eef_abs_for_file(
            df=sample_df,
            fk=fk,
            action_key=args.action_key,
            joint_indices=args.joint_indices,
            gripper_action_index=args.gripper_action_index,
            show_progress=args.show_progress,
            progress_desc="[DRY-RUN] FK sample",
        )

        print("[DRY-RUN] Validation succeeded.")
        print(f"  dataset_root={dataset_root}")
        print(f"  total_data_files={len(data_files)}")
        print(f"  sample_file={data_files[0]}")
        print(f"  output_key={args.output_key}")
        print(f"  sample_shape={sample_eef_abs.shape}")
        print(f"  sample_first_row={sample_eef_abs[0].tolist()}")
        print(f"  sample_rotvec_abs_max={float(np.max(np.abs(sample_eef_abs[:, 3:6]))):.6f}")
        if alignment_result is not None:
            print(f"  fk_alignment_convention={alignment_result.euler_convention}")
        return

    _prepare_output_dir(dataset_root=dataset_root, output_root=output_root, overwrite_output=args.overwrite_output)

    if fk is None:
        raise RuntimeError("Internal error: FK engine is not initialized for conversion.")

    total_files, total_rows, stats_payload = _process_dataset(
        output_root=output_root,
        fk=fk,
        action_key=args.action_key,
        joint_indices=args.joint_indices,
        gripper_action_index=args.gripper_action_index,
        output_key=args.output_key,
        recompute_stats=args.recompute_stats,
        show_progress=args.show_progress,
    )

    output_info_path = output_root / "meta" / "info.json"
    _update_info_json(output_info_path, args.output_key)

    if args.recompute_stats:
        if stats_payload is None:
            raise ValueError("Internal error: stats payload is missing while recompute-stats is enabled.")
        output_stats_path = output_root / "meta" / "stats.json"
        _update_stats_json(output_stats_path, args.output_key, stats_payload)

    print("Completed EEF absolute-action augmentation.")
    print(f"  input_root: {dataset_root}")
    print(f"  output_root: {output_root}")
    print(f"  output_key: {args.output_key}")
    print(f"  feature_layout: {FEATURE_NAMES}")
    print(f"  files_processed: {total_files}")
    print(f"  rows_processed: {total_rows}")
    print(f"  stats_recomputed: {args.recompute_stats}")
    print(f"  fk_alignment_checked: {args.validate_fk_alignment}")
    print(f"  run_conversion: {args.run_conversion}")
    if alignment_result is not None:
        print(f"  fk_alignment_convention: {alignment_result.euler_convention}")


if __name__ == "__main__":
    main()
