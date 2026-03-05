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


DEFAULT_DATASET_ROOT = Path("/mnt/project_rlinf/jlchen/datasets/AgiBot_merge")
DEFAULT_SOURCE_KEY = "actions.end.orientation"


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


def _normalize_quat_xyzw(batch_quat: np.ndarray, key: str) -> np.ndarray:
    if batch_quat.ndim != 2 or batch_quat.shape[1] != 4:
        raise ValueError(f"{key} should have shape [N, 4], got {batch_quat.shape}.")
    norms = np.linalg.norm(batch_quat, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{key} contains near-zero-norm quaternion samples.")
    return batch_quat / norms


def _quat8_to_rotvec6(batch_quat8: np.ndarray, key: str) -> np.ndarray:
    if batch_quat8.ndim != 2 or batch_quat8.shape[1] != 8:
        raise ValueError(f"{key} should have shape [N, 8], got {batch_quat8.shape}.")
    left_xyzw = _normalize_quat_xyzw(batch_quat8[:, :4], f"{key}.left")
    right_xyzw = _normalize_quat_xyzw(batch_quat8[:, 4:], f"{key}.right")
    left_rotvec = R.from_quat(left_xyzw).as_rotvec().astype(np.float32)
    right_rotvec = R.from_quat(right_xyzw).as_rotvec().astype(np.float32)
    return np.concatenate([left_rotvec, right_rotvec], axis=1).astype(np.float32)


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
    in_place_replace: bool,
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
        file_iter = tqdm(files, total=len(files), desc="[CONVERT] Orientation", unit="file")

    for file_path in file_iter:
        df = pd.read_parquet(file_path)
        if max_rows_per_file is not None and max_rows_per_file > 0:
            df = df.head(max_rows_per_file)
        if source_key not in df.columns:
            raise KeyError(f"Required key '{source_key}' missing in {file_path}")
        if (not in_place_replace) and output_key in df.columns:
            raise ValueError(f"Output key '{output_key}' already exists in {file_path}")

        source_arr = _to_2d_array(df[source_key], source_key)
        if source_arr.shape[1] != 8:
            raise ValueError(f"Expected {source_key} dim=8, got {source_arr.shape} in {file_path}")
        rotvec_arr = _quat8_to_rotvec6(source_arr, source_key)

        row_count = len(df)
        total_rows += row_count
        total_files += 1
        stats_chunks.append(rotvec_arr)

        if in_place_replace:
            df[source_key] = [rotvec_arr[i] for i in range(row_count)]
        else:
            df[output_key] = [rotvec_arr[i] for i in range(row_count)]

        if not dry_run:
            df.to_parquet(file_path, index=False)

    if total_rows == 0:
        raise ValueError("All parquet files are empty. Nothing was processed.")
    return total_files, total_rows, np.concatenate(stats_chunks, axis=0)


def _update_info_json(
    output_root: Path,
    source_key: str,
    output_key: str,
    in_place_replace: bool,
) -> None:
    info_path = output_root / "meta" / "info.json"
    info = _load_json(info_path)
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Invalid info.json features in {info_path}")
    if source_key not in features:
        raise KeyError(f"{source_key} not found in info.json features")
    if (not in_place_replace) and output_key in features:
        raise ValueError(f"{output_key} already exists in info.json features")

    target_key = source_key if in_place_replace else output_key
    features[target_key] = {
        "dtype": "float32",
        "shape": [6],
        "names": {
            "motors": [
                "left_rx",
                "left_ry",
                "left_rz",
                "right_rx",
                "right_ry",
                "right_rz",
            ]
        },
    }
    info["features"] = features
    _save_json(info_path, info)


def _update_modality_json(
    output_root: Path,
    source_key: str,
    output_key: str,
    in_place_replace: bool,
) -> None:
    modality_path = output_root / "meta" / "modality.json"
    if not modality_path.exists():
        return
    modality = _load_json(modality_path)
    action = modality.get("action")
    if not isinstance(action, dict):
        raise ValueError("modality.json must contain top-level 'action' dict.")

    target_key = source_key if in_place_replace else output_key
    for side, start, end in [("left", 0, 3), ("right", 3, 6)]:
        key = f"end_orientation_{side}"
        if key in action and isinstance(action[key], dict):
            node = dict(action[key])
            node["start"] = start
            node["end"] = end
            node["original_key"] = target_key
            node["rotation_type"] = "axis_angle"
            node["dtype"] = "float32"
            action[key] = node
        else:
            action[key] = {
                "start": start,
                "end": end,
                "original_key": target_key,
                "rotation_type": "axis_angle",
                "dtype": "float32",
            }

    if not in_place_replace:
        # Add explicit rotvec aliases for clarity.
        action["end_orientation_left_rotvec"] = {
            "start": 0,
            "end": 3,
            "original_key": target_key,
            "rotation_type": "axis_angle",
            "dtype": "float32",
        }
        action["end_orientation_right_rotvec"] = {
            "start": 3,
            "end": 6,
            "original_key": target_key,
            "rotation_type": "axis_angle",
            "dtype": "float32",
        }

    modality["action"] = action
    _save_json(modality_path, modality)


def _update_stats_files(output_root: Path, source_key: str, output_key: str, in_place_replace: bool, stats_payload: dict[str, list[float]]) -> None:
    target_key = source_key if in_place_replace else output_key
    for stats_name in ["stats.json", "stats_gr00t.json"]:
        sp = output_root / "meta" / stats_name
        if not sp.exists():
            continue
        stats = _load_json(sp)
        if not isinstance(stats, dict):
            raise ValueError(f"Invalid stats json: {sp}")
        stats[target_key] = stats_payload
        _save_json(sp, stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert AgiBot action orientation from quaternion(8) to rotvec(6): "
            "[left_xyzw,right_xyzw] -> [left_rxryrz,right_rxryrz]."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--source-key", type=str, default=DEFAULT_SOURCE_KEY)
    parser.add_argument(
        "--output-key",
        type=str,
        default=DEFAULT_SOURCE_KEY,
        help="Output key. Default equals source-key for in-place replacement.",
    )
    parser.add_argument("--in-place", action="store_true", help="Modify dataset-root directly.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output dataset root when not --in-place (default: <dataset-root>_orientation_rotvec).",
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

    source_key = args.source_key
    output_key = args.output_key
    in_place_replace = output_key == source_key

    if args.in_place:
        if args.output_root is not None:
            raise ValueError("Do not set --output-root with --in-place.")
        output_root = dataset_root
    else:
        if args.output_root is None:
            output_root = dataset_root.parent / f"{dataset_root.name}_orientation_rotvec"
        else:
            output_root = args.output_root.resolve()
        if output_root == dataset_root:
            raise ValueError("output-root must be different from dataset-root unless --in-place is set.")

    info_path = dataset_root / "meta" / "info.json"
    info = _load_json(info_path)
    features = info.get("features", {})
    if source_key not in features:
        raise KeyError(f"{source_key} not found in source dataset features.")
    if (not in_place_replace) and output_key in features:
        raise ValueError(f"{output_key} already exists in source dataset features.")

    if not args.dry_run:
        _prepare_output_dir(dataset_root, output_root, args.overwrite_output, args.in_place)

    total_files, total_rows, all_rotvec = _process_data_files(
        dataset_root=dataset_root if args.dry_run else output_root,
        source_key=source_key,
        output_key=output_key,
        in_place_replace=in_place_replace,
        dry_run=args.dry_run,
        show_progress=args.show_progress,
        max_files=args.dry_run_max_files if args.dry_run else None,
        max_rows_per_file=args.dry_run_max_rows if args.dry_run else None,
    )
    stats_payload = _stats_from_array(all_rotvec)

    if not args.dry_run:
        _update_info_json(output_root, source_key, output_key, in_place_replace)
        _update_modality_json(output_root, source_key, output_key, in_place_replace)
        _update_stats_files(output_root, source_key, output_key, in_place_replace, stats_payload)

    print("Completed orientation conversion.")
    print(f"  dataset_root: {dataset_root}")
    print(f"  output_root: {output_root}")
    print(f"  source_key: {source_key}")
    print(f"  output_key: {output_key}")
    print(f"  in_place_replace: {in_place_replace}")
    print(f"  dry_run: {args.dry_run}")
    print(f"  files_processed: {total_files}")
    print(f"  rows_processed: {total_rows}")
    print(f"  rotvec_dim: {all_rotvec.shape[1]}")
    print(f"  first_row_rotvec: {all_rotvec[0].tolist()}")


if __name__ == "__main__":
    main()
