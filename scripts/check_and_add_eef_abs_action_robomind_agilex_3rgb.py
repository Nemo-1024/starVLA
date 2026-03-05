#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm


DEFAULT_DATASET_ROOT = Path("/mnt/project_rlinf/jlchen/datasets/robomind_agilex_3rgb")
DEFAULT_LEFT_SOURCE_KEY = "actions.end_effector_left"
DEFAULT_RIGHT_SOURCE_KEY = "actions.end_effector_right"
DEFAULT_LEFT_OUTPUT_KEY = "actions.end_effector_left_rotvec"
DEFAULT_RIGHT_OUTPUT_KEY = "actions.end_effector_right_rotvec"
DEFAULT_LEFT_OBS_ROTVEC_KEY = "observation.states.end_effector_left_rotvec"
DEFAULT_RIGHT_OBS_ROTVEC_KEY = "observation.states.end_effector_right_rotvec"


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


def _prepare_output_dir(dataset_root: Path, output_root: Path, overwrite_output: bool, in_place: bool) -> None:
    if in_place:
        return
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


def _normalize_quat_xyzw(batch_quat: np.ndarray, key: str, eps: float) -> np.ndarray:
    if batch_quat.ndim != 2 or batch_quat.shape[1] != 4:
        raise ValueError(f"{key} should have shape [N, 4], got {batch_quat.shape}.")
    norms = np.linalg.norm(batch_quat, axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError(f"{key} contains near-zero-norm quaternion samples (eps={eps}).")
    return (batch_quat / norms).astype(np.float32)


def _eef_quat_to_pose_rotvec(batch_eef: np.ndarray, key: str, eps: float) -> np.ndarray:
    # Expected layout: [x, y, z, qx, qy, qz, qw] with possibly non-unit quaternions.
    if batch_eef.ndim != 2 or batch_eef.shape[1] != 7:
        raise ValueError(f"{key} should have shape [N, 7], got {batch_eef.shape}.")
    pos = batch_eef[:, 0:3].astype(np.float32)
    quat_xyzw = _normalize_quat_xyzw(batch_eef[:, 3:7], key=f"{key}[3:7]", eps=eps)
    rotvec = R.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
    return np.concatenate([pos, rotvec], axis=1).astype(np.float32)


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


def _run_quaternion_detection(
    dataset_root: Path,
    left_source_key: str,
    right_source_key: str,
    left_obs_rotvec_key: str,
    right_obs_rotvec_key: str,
    check_rows: int,
    quat_eps: float,
    show_progress: bool,
) -> None:
    files = _collect_data_files(dataset_root)
    left_chunks: list[np.ndarray] = []
    right_chunks: list[np.ndarray] = []
    left_obs_chunks: list[np.ndarray] = []
    right_obs_chunks: list[np.ndarray] = []
    remaining = check_rows if check_rows > 0 else None

    file_iter = files
    if show_progress:
        file_iter = tqdm(files, total=len(files), desc="[CHECK] Detect quaternion layout", unit="file")

    for file_path in file_iter:
        cols = [left_source_key, right_source_key]
        has_obs_rotvec = False
        df_head = pd.read_parquet(file_path, columns=cols).head(1)
        if left_obs_rotvec_key in df_head.columns and right_obs_rotvec_key in df_head.columns:
            has_obs_rotvec = True
        if has_obs_rotvec:
            cols = [left_source_key, right_source_key, left_obs_rotvec_key, right_obs_rotvec_key]

        df = pd.read_parquet(file_path, columns=cols)
        if len(df) == 0:
            continue

        left = _to_2d_array(df[left_source_key], left_source_key)
        right = _to_2d_array(df[right_source_key], right_source_key)

        if remaining is not None:
            take = min(remaining, left.shape[0])
            left = left[:take]
            right = right[:take]
            if has_obs_rotvec:
                left_obs = _to_2d_array(df[left_obs_rotvec_key], left_obs_rotvec_key)[:take]
                right_obs = _to_2d_array(df[right_obs_rotvec_key], right_obs_rotvec_key)[:take]
            remaining -= take
        elif has_obs_rotvec:
            left_obs = _to_2d_array(df[left_obs_rotvec_key], left_obs_rotvec_key)
            right_obs = _to_2d_array(df[right_obs_rotvec_key], right_obs_rotvec_key)

        left_chunks.append(left)
        right_chunks.append(right)
        if has_obs_rotvec:
            left_obs_chunks.append(left_obs)
            right_obs_chunks.append(right_obs)

        if remaining is not None and remaining <= 0:
            break

    if not left_chunks:
        raise ValueError("No rows found for quaternion detection.")

    left_all = np.concatenate(left_chunks, axis=0)
    right_all = np.concatenate(right_chunks, axis=0)
    if left_all.shape[1] != 7 or right_all.shape[1] != 7:
        raise ValueError(
            f"Expected source dims [N,7]. got left={left_all.shape}, right={right_all.shape}. "
            "This detector expects [x,y,z,qx,qy,qz,qw]."
        )

    norms = np.concatenate([np.linalg.norm(left_all[:, 3:7], axis=1), np.linalg.norm(right_all[:, 3:7], axis=1)])
    ratio_close_1_1e2 = float((np.abs(norms - 1.0) < 1e-2).mean())
    ratio_close_1_5e2 = float((np.abs(norms - 1.0) < 5e-2).mean())

    print("[CHECK] Source orientation layout detection")
    print("  hypothesis: source[3:7] are non-unit quaternions (xyzw) and should be normalized before conversion")
    print(f"  rows_checked: {norms.shape[0]}")
    print(
        f"  quat_norm mean={float(norms.mean()):.6f}, std={float(norms.std()):.6f}, "
        f"p01={float(np.quantile(norms,0.01)):.6f}, p50={float(np.quantile(norms,0.50)):.6f}, "
        f"p99={float(np.quantile(norms,0.99)):.6f}"
    )
    print(f"  ratio(|norm-1|<1e-2)={ratio_close_1_1e2:.6f}")
    print(f"  ratio(|norm-1|<5e-2)={ratio_close_1_5e2:.6f}")

    if left_obs_chunks and right_obs_chunks:
        left_obs_all = np.concatenate(left_obs_chunks, axis=0)
        right_obs_all = np.concatenate(right_obs_chunks, axis=0)
        if left_obs_all.shape[1] >= 3 and right_obs_all.shape[1] >= 3:
            left_q = _normalize_quat_xyzw(left_all[:, 3:7], "left_source_quat", quat_eps)
            right_q = _normalize_quat_xyzw(right_all[:, 3:7], "right_source_quat", quat_eps)
            left_pred = R.from_quat(left_q).as_rotvec().astype(np.float32)
            right_pred = R.from_quat(right_q).as_rotvec().astype(np.float32)
            left_err = np.linalg.norm(left_pred - left_obs_all[:, :3], axis=1)
            right_err = np.linalg.norm(right_pred - right_obs_all[:, :3], axis=1)
            all_err = np.concatenate([left_err, right_err], axis=0)
            print("  cross-check with observation.states.*_rotvec:")
            print(
                f"    rotvec_diff mean={float(all_err.mean()):.8f}, "
                f"p95={float(np.quantile(all_err,0.95)):.8f}, max={float(all_err.max()):.8f}"
            )

    print("  conclusion: convert by normalizing source[3:7] as quaternion then as_rotvec.")


def _process_data_files(
    dataset_root: Path,
    left_source_key: str,
    right_source_key: str,
    left_output_key: str,
    right_output_key: str,
    dry_run: bool,
    show_progress: bool,
    quat_eps: float,
    max_files: int | None = None,
    max_rows_per_file: int | None = None,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    files = _collect_data_files(dataset_root)
    if max_files is not None and max_files > 0:
        files = files[:max_files]

    total_files = 0
    total_rows = 0
    left_stats_chunks: list[np.ndarray] = []
    right_stats_chunks: list[np.ndarray] = []

    file_iter = files
    if show_progress:
        file_iter = tqdm(files, total=len(files), desc="[CONVERT] end_effector -> rotvec", unit="file")

    for file_path in file_iter:
        df = pd.read_parquet(file_path)
        if max_rows_per_file is not None and max_rows_per_file > 0:
            df = df.head(max_rows_per_file)

        if left_source_key not in df.columns or right_source_key not in df.columns:
            raise KeyError(f"Missing source key in {file_path}.")
        if left_output_key in df.columns or right_output_key in df.columns:
            raise ValueError(
                f"Output key already exists in {file_path}: "
                f"left_exists={left_output_key in df.columns}, right_exists={right_output_key in df.columns}"
            )

        left_src = _to_2d_array(df[left_source_key], left_source_key)
        right_src = _to_2d_array(df[right_source_key], right_source_key)
        left_out = _eef_quat_to_pose_rotvec(left_src, left_source_key, quat_eps)
        right_out = _eef_quat_to_pose_rotvec(right_src, right_source_key, quat_eps)

        row_count = len(df)
        total_files += 1
        total_rows += row_count
        left_stats_chunks.append(left_out)
        right_stats_chunks.append(right_out)

        df[left_output_key] = [left_out[i] for i in range(row_count)]
        df[right_output_key] = [right_out[i] for i in range(row_count)]
        if not dry_run:
            df.to_parquet(file_path, index=False)

    if total_rows == 0:
        raise ValueError("All parquet files are empty. Nothing was processed.")
    return (
        total_files,
        total_rows,
        np.concatenate(left_stats_chunks, axis=0),
        np.concatenate(right_stats_chunks, axis=0),
    )


def _update_info_json(output_root: Path, left_output_key: str, right_output_key: str) -> None:
    info_path = output_root / "meta" / "info.json"
    info = _load_json(info_path)
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Invalid info.json features in {info_path}")
    for key in [left_output_key, right_output_key]:
        if key in features:
            raise ValueError(f"{key} already exists in info.json features.")
        features[key] = {
            "dtype": "float32",
            "shape": [6],
            "names": {"motors": ["x", "y", "z", "rx", "ry", "rz"]},
        }
    info["features"] = features
    _save_json(info_path, info)


def _update_modality_json(output_root: Path, left_output_key: str, right_output_key: str) -> None:
    modality_path = output_root / "meta" / "modality.json"
    if not modality_path.exists():
        return

    modality = _load_json(modality_path)
    action = modality.get("action")
    if not isinstance(action, dict):
        raise ValueError("modality.json must contain top-level 'action' dict.")

    additions = {
        "end_position_left_rotvec_action": {
            "start": 0,
            "end": 3,
            "original_key": left_output_key,
            "absolute": True,
            "dtype": "float32",
        },
        "end_orientation_left_rotvec_action": {
            "start": 3,
            "end": 6,
            "rotation_type": "axis_angle",
            "original_key": left_output_key,
            "absolute": True,
            "dtype": "float32",
        },
        "end_position_right_rotvec_action": {
            "start": 0,
            "end": 3,
            "original_key": right_output_key,
            "absolute": True,
            "dtype": "float32",
        },
        "end_orientation_right_rotvec_action": {
            "start": 3,
            "end": 6,
            "rotation_type": "axis_angle",
            "original_key": right_output_key,
            "absolute": True,
            "dtype": "float32",
        },
    }

    duplicates = [k for k in additions if k in action]
    if duplicates:
        raise ValueError(f"modality.json action keys already exist: {duplicates}")
    action.update(additions)
    modality["action"] = action
    _save_json(modality_path, modality)


def _update_stats_files(
    output_root: Path,
    left_output_key: str,
    right_output_key: str,
    left_stats: dict[str, list[float]],
    right_stats: dict[str, list[float]],
) -> None:
    for stats_name in ["stats.json", "stats_gr00t.json"]:
        stats_path = output_root / "meta" / stats_name
        if not stats_path.exists():
            continue
        stats = _load_json(stats_path)
        if not isinstance(stats, dict):
            raise ValueError(f"Invalid stats file: {stats_path}")
        if left_output_key in stats or right_output_key in stats:
            raise ValueError(
                f"Output key already exists in {stats_path}: "
                f"left_exists={left_output_key in stats}, right_exists={right_output_key in stats}"
            )
        stats[left_output_key] = left_stats
        stats[right_output_key] = right_stats
        _save_json(stats_path, stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect quaternion layout in actions.end_effector_{left,right}, then convert to "
            "position+rotvec and save as new keys."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--left-source-key", type=str, default=DEFAULT_LEFT_SOURCE_KEY)
    parser.add_argument("--right-source-key", type=str, default=DEFAULT_RIGHT_SOURCE_KEY)
    parser.add_argument("--left-output-key", type=str, default=DEFAULT_LEFT_OUTPUT_KEY)
    parser.add_argument("--right-output-key", type=str, default=DEFAULT_RIGHT_OUTPUT_KEY)
    parser.add_argument("--left-obs-rotvec-key", type=str, default=DEFAULT_LEFT_OBS_ROTVEC_KEY)
    parser.add_argument("--right-obs-rotvec-key", type=str, default=DEFAULT_RIGHT_OBS_ROTVEC_KEY)
    parser.add_argument("--quat-eps", type=float, default=1e-12)
    parser.add_argument("--check-rows", type=int, default=100000)

    parser.add_argument("--in-place", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output dataset root when not --in-place (default: <dataset-root>_eef_rotvec_action).",
    )
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dry-run-max-files",
        type=int,
        default=1,
        help="When --dry-run is set, process at most this many files (default: 1).",
    )
    parser.add_argument(
        "--dry-run-max-rows",
        type=int,
        default=16,
        help="When --dry-run is set, process at most this many rows per file (default: 16).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if args.left_output_key == args.right_output_key:
        raise ValueError("left-output-key and right-output-key must be different.")

    info = _load_json(dataset_root / "meta" / "info.json")
    features = info.get("features", {})
    for key in [args.left_source_key, args.right_source_key]:
        if key not in features:
            raise KeyError(f"{key} not found in source dataset features.")
    for key in [args.left_output_key, args.right_output_key]:
        if key in features:
            raise ValueError(f"{key} already exists in source dataset features.")

    _run_quaternion_detection(
        dataset_root=dataset_root,
        left_source_key=args.left_source_key,
        right_source_key=args.right_source_key,
        left_obs_rotvec_key=args.left_obs_rotvec_key,
        right_obs_rotvec_key=args.right_obs_rotvec_key,
        check_rows=args.check_rows,
        quat_eps=args.quat_eps,
        show_progress=args.show_progress,
    )

    if args.in_place:
        if args.output_root is not None:
            raise ValueError("Do not set --output-root with --in-place.")
        output_root = dataset_root
    else:
        if args.output_root is None:
            output_root = dataset_root.parent / f"{dataset_root.name}_eef_rotvec_action"
        else:
            output_root = args.output_root.resolve()
        if output_root == dataset_root:
            raise ValueError("output-root must differ from dataset-root unless --in-place is set.")

    if not args.dry_run:
        _prepare_output_dir(dataset_root, output_root, args.overwrite_output, args.in_place)

    total_files, total_rows, left_all, right_all = _process_data_files(
        dataset_root=dataset_root if args.dry_run else output_root,
        left_source_key=args.left_source_key,
        right_source_key=args.right_source_key,
        left_output_key=args.left_output_key,
        right_output_key=args.right_output_key,
        dry_run=args.dry_run,
        show_progress=args.show_progress,
        quat_eps=args.quat_eps,
        max_files=args.dry_run_max_files if args.dry_run else None,
        max_rows_per_file=args.dry_run_max_rows if args.dry_run else None,
    )
    left_stats = _stats_from_array(left_all)
    right_stats = _stats_from_array(right_all)

    if not args.dry_run:
        _update_info_json(output_root, args.left_output_key, args.right_output_key)
        _update_modality_json(output_root, args.left_output_key, args.right_output_key)
        _update_stats_files(output_root, args.left_output_key, args.right_output_key, left_stats, right_stats)

    print("Completed conversion from end_effector quaternion-like action to rotvec action keys.")
    print(f"  dataset_root: {dataset_root}")
    print(f"  output_root: {output_root}")
    print(f"  left_source_key: {args.left_source_key}")
    print(f"  right_source_key: {args.right_source_key}")
    print(f"  left_output_key: {args.left_output_key}")
    print(f"  right_output_key: {args.right_output_key}")
    print(f"  dry_run: {args.dry_run}")
    print(f"  files_processed: {total_files}")
    print(f"  rows_processed: {total_rows}")
    print(f"  first_left_row: {left_all[0].tolist()}")
    print(f"  first_right_row: {right_all[0].tolist()}")


if __name__ == "__main__":
    main()
