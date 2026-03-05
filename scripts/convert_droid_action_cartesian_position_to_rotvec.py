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


DEFAULT_DATASET_ROOT = Path("/mnt/project_rlinf/jlchen/datasets/droid_1.0.1")
DEFAULT_SOURCE_KEY = "action.cartesian_position"
DEFAULT_OUTPUT_KEY = "action.cartesian_position_rotvec"


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


def _cartesian_position_to_rotvec(batch: np.ndarray, key: str) -> np.ndarray:
    if batch.ndim != 2 or batch.shape[1] != 6:
        raise ValueError(f"{key} should have shape [N, 6], got {batch.shape}.")
    euler_xyz = batch[:, 3:6].astype(np.float64)
    rotvec = R.from_euler("xyz", euler_xyz).as_rotvec().astype(np.float32)
    return rotvec


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


def _process_data_files(
    dataset_root: Path,
    source_key: str,
    output_key: str,
    dry_run: bool,
    show_progress: bool,
    max_files: int | None = None,
    max_rows_per_file: int | None = None,
) -> tuple[int, int, np.ndarray]:
    files = _collect_data_files(dataset_root)
    if max_files is not None and max_files > 0:
        files = files[:max_files]

    total_files = 0
    total_rows = 0
    stats_chunks: list[np.ndarray] = []

    file_iter = files
    if show_progress:
        file_iter = tqdm(files, total=len(files), desc="[CONVERT] action.cartesian_position", unit="file")

    for file_path in file_iter:
        df = pd.read_parquet(file_path)
        if max_rows_per_file is not None and max_rows_per_file > 0:
            df = df.head(max_rows_per_file)
        if source_key not in df.columns:
            raise KeyError(f"Required key '{source_key}' missing in {file_path}")
        if output_key in df.columns:
            raise ValueError(f"Output key '{output_key}' already exists in {file_path}")

        source_arr = _to_2d_array(df[source_key], source_key)
        rotvec_arr = _cartesian_position_to_rotvec(source_arr, source_key)

        row_count = len(df)
        total_rows += row_count
        total_files += 1
        stats_chunks.append(rotvec_arr)

        df[output_key] = [rotvec_arr[i] for i in range(row_count)]
        if not dry_run:
            df.to_parquet(file_path, index=False)

    if total_rows == 0:
        raise ValueError("All parquet files are empty. Nothing was processed.")
    return total_files, total_rows, np.concatenate(stats_chunks, axis=0)


def _update_info_json(output_root: Path, output_key: str) -> None:
    info_path = output_root / "meta" / "info.json"
    info = _load_json(info_path)
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Invalid info.json features in {info_path}")
    if output_key in features:
        raise ValueError(f"{output_key} already exists in info.json features.")
    features[output_key] = {
        "dtype": "float32",
        "shape": [3],
        "names": {"motors": ["rx", "ry", "rz"]},
    }
    info["features"] = features
    _save_json(info_path, info)


def _update_modality_json(output_root: Path, output_key: str) -> None:
    modality_path = output_root / "meta" / "modality.json"
    if not modality_path.exists():
        return
    modality = _load_json(modality_path)
    action = modality.get("action")
    if not isinstance(action, dict):
        raise ValueError("modality.json must contain top-level 'action' dict.")
    key = "eef_orientation_position_rotvec"
    if key in action:
        raise ValueError(f"modality.json action key '{key}' already exists.")
    action[key] = {
        "start": 0,
        "end": 3,
        "rotation_type": "axis_angle",
        "original_key": output_key,
        "dtype": "float32",
    }
    modality["action"] = action
    _save_json(modality_path, modality)


def _update_stats_files(output_root: Path, output_key: str, stats_payload: dict[str, list[float]]) -> None:
    for stats_name in ["stats.json", "stats_gr00t.json"]:
        stats_path = output_root / "meta" / stats_name
        if not stats_path.exists():
            continue
        stats = _load_json(stats_path)
        if not isinstance(stats, dict):
            raise ValueError(f"Invalid stats json: {stats_path}")
        if output_key in stats:
            raise ValueError(f"{output_key} already exists in {stats_path}")
        stats[output_key] = stats_payload
        _save_json(stats_path, stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert action.cartesian_position orientation (Euler xyz in indices 3:6) "
            "to rotvec and save to a new key."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--source-key", type=str, default=DEFAULT_SOURCE_KEY)
    parser.add_argument("--output-key", type=str, default=DEFAULT_OUTPUT_KEY)
    parser.add_argument("--in-place", action="store_true", help="Modify dataset-root directly.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output dataset root when not --in-place (default: <dataset-root>_action_rotvec).",
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
        help="When --dry-run is set, process at most this many parquet files (default: 1).",
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
    if args.source_key == args.output_key:
        raise ValueError("source-key and output-key must be different; this script adds a new key.")

    info = _load_json(dataset_root / "meta" / "info.json")
    features = info.get("features", {})
    if args.source_key not in features:
        raise KeyError(f"{args.source_key} not found in source dataset features.")
    if args.output_key in features:
        raise ValueError(f"{args.output_key} already exists in source dataset features.")

    if args.in_place:
        if args.output_root is not None:
            raise ValueError("Do not set --output-root with --in-place.")
        output_root = dataset_root
    else:
        if args.output_root is None:
            output_root = dataset_root.parent / f"{dataset_root.name}_action_rotvec"
        else:
            output_root = args.output_root.resolve()
        if output_root == dataset_root:
            raise ValueError("output-root must be different from dataset-root unless --in-place is set.")

    if not args.dry_run:
        _prepare_output_dir(dataset_root, output_root, args.overwrite_output, args.in_place)

    total_files, total_rows, all_rotvec = _process_data_files(
        dataset_root=dataset_root if args.dry_run else output_root,
        source_key=args.source_key,
        output_key=args.output_key,
        dry_run=args.dry_run,
        show_progress=args.show_progress,
        max_files=args.dry_run_max_files if args.dry_run else None,
        max_rows_per_file=args.dry_run_max_rows if args.dry_run else None,
    )
    stats_payload = _stats_from_array(all_rotvec)

    if not args.dry_run:
        _update_info_json(output_root, args.output_key)
        _update_modality_json(output_root, args.output_key)
        _update_stats_files(output_root, args.output_key, stats_payload)

    print("Completed droid action orientation->rotvec conversion.")
    print(f"  dataset_root: {dataset_root}")
    print(f"  output_root: {output_root}")
    print(f"  source_key: {args.source_key}")
    print(f"  output_key: {args.output_key}")
    print(f"  dry_run: {args.dry_run}")
    print(f"  files_processed: {total_files}")
    print(f"  rows_processed: {total_rows}")
    print(f"  rotvec_dim: {all_rotvec.shape[1]}")
    print(f"  first_row_rotvec: {all_rotvec[0].tolist()}")


if __name__ == "__main__":
    main()
