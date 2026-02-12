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

def _ensure_starvla_on_path() -> None:
    if importlib.util.find_spec("starVLA") is not None:
        return
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root
    if candidate.exists():
        sys.path.insert(0, str(candidate))
    if importlib.util.find_spec("starVLA") is None:
        raise ModuleNotFoundError(
            "Cannot import 'starVLA'. Please run from repo root or add starVLA package path to PYTHONPATH."
        )


_ensure_starvla_on_path()

from starVLA.dataloader.gr00t_lerobot.transform.state_action import RotationTransform


def _ensure_lerobot_on_path() -> None:
    if importlib.util.find_spec("lerobot") is not None:
        return

    candidates = []
    env_path = Path(str(Path.cwd()))
    candidates.append(env_path / ".." / "lerobot" / "src")
    candidates.append(Path(__file__).resolve().parents[2] / "lerobot" / "src")
    candidates.append(Path("/mnt/project_rlinf/jlchen/code/lerobot/src"))

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            if importlib.util.find_spec("lerobot") is not None:
                return

    raise ModuleNotFoundError(
        "Cannot import 'lerobot'. Please ensure lerobot is installed or add its src path to PYTHONPATH."
    )


_ensure_lerobot_on_path()

from lerobot.datasets.compute_stats import DEFAULT_QUANTILES, aggregate_stats, get_feature_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_stats, write_stats


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def _normalize_quaternion_order(batch: np.ndarray, order: str) -> np.ndarray:
    order = order.lower()
    if order == "wxyz":
        return batch
    if order == "xyzw":
        return batch[:, [3, 0, 1, 2]]
    raise ValueError(f"Invalid quaternion_order '{order}'. Expected 'wxyz' or 'xyzw'.")


def _to_2d_array(column_values: pd.Series, key: str) -> np.ndarray:
    rows = []
    for idx, value in enumerate(column_values):
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        rows.append(arr.reshape(-1))
    try:
        stacked = np.stack(rows, axis=0)
    except Exception as exc:
        raise ValueError(f"Failed to stack values for key '{key}' into a consistent array.") from exc
    return stacked


def _convert_source_to_rot6d(
    df: pd.DataFrame,
    source_cfg: dict[str, Any],
    row_count: int,
) -> np.ndarray:
    input_key = source_cfg["input_key"]
    from_rep = source_cfg["from_rep"]
    quaternion_order = source_cfg.get("quaternion_order", "wxyz")
    degrees = bool(source_cfg.get("degrees", False))
    indices = source_cfg.get("indices")

    if input_key not in df.columns:
        raise KeyError(f"Input key '{input_key}' not found in parquet columns.")

    source_arr = _to_2d_array(df[input_key], input_key)
    if source_arr.shape[0] != row_count:
        raise ValueError(
            f"Row count mismatch for key '{input_key}'. "
            f"Expected {row_count}, got {source_arr.shape[0]}."
        )

    if indices is not None:
        if not isinstance(indices, list) or len(indices) == 0:
            raise ValueError(f"'indices' for key '{input_key}' must be a non-empty list of integers.")
        if any((not isinstance(i, int)) or isinstance(i, bool) for i in indices):
            raise ValueError(f"'indices' for key '{input_key}' must contain integers only.")
        if len(set(indices)) != len(indices):
            raise ValueError(f"'indices' for key '{input_key}' contains duplicates: {indices}")
        feat_dim = source_arr.shape[1]
        if any(i < 0 or i >= feat_dim for i in indices):
            raise ValueError(
                f"'indices' for key '{input_key}' out of range for dim={feat_dim}. got={indices}"
            )
        source_arr = source_arr[:, indices]

    if from_rep == "rotation_6d":
        if source_arr.shape[1] != 6:
            raise ValueError(f"Input key '{input_key}' claims rotation_6d but has dim={source_arr.shape[1]}.")
        return source_arr.astype(np.float32)

    if from_rep == "quaternion":
        if source_arr.shape[1] != 4:
            raise ValueError(f"Quaternion key '{input_key}' must have dim=4, got {source_arr.shape[1]}.")
        source_arr = _normalize_quaternion_order(source_arr, quaternion_order)
        norms = np.linalg.norm(source_arr, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError(f"Quaternion key '{input_key}' contains near-zero norm samples.")
        source_arr = source_arr / norms
        source_tensor = torch.from_numpy(source_arr)
    elif from_rep.startswith("euler_angles"):
        if source_arr.shape[1] != 3:
            raise ValueError(f"Euler key '{input_key}' must have dim=3, got {source_arr.shape[1]}.")
        if degrees:
            source_arr = np.deg2rad(source_arr)
        source_tensor = torch.from_numpy(source_arr)
    elif from_rep == "axis_angle":
        if source_arr.shape[1] != 3:
            raise ValueError(f"Axis-angle key '{input_key}' must have dim=3, got {source_arr.shape[1]}.")
        source_tensor = torch.from_numpy(source_arr)
    elif from_rep == "matrix":
        if source_arr.shape[1] != 9:
            raise ValueError(f"Matrix key '{input_key}' must flatten to dim=9, got {source_arr.shape[1]}.")
        source_tensor = torch.from_numpy(source_arr.reshape(-1, 3, 3))
    else:
        raise ValueError(
            f"Unsupported from_rep '{from_rep}' for key '{input_key}'. "
            "Use quaternion, euler_angles_*, axis_angle, matrix, or rotation_6d."
        )

    transformer = RotationTransform(from_rep=from_rep, to_rep="rotation_6d")
    converted = transformer.forward(source_tensor).cpu().numpy().astype(np.float32)
    if converted.ndim != 2 or converted.shape[1] != 6:
        raise ValueError(f"Converted key '{input_key}' has invalid shape {converted.shape}, expected [N, 6].")
    return converted


def _validate_mapping(mapping: dict[str, Any]) -> None:
    for modality in ("state", "action"):
        if modality not in mapping:
            raise ValueError(f"mapping-json must contain '{modality}'.")
        node = mapping[modality]
        if "output_key" not in node:
            raise ValueError(f"mapping-json.{modality}.output_key is required.")
        if "sources" not in node or not isinstance(node["sources"], list) or len(node["sources"]) == 0:
            raise ValueError(f"mapping-json.{modality}.sources must be a non-empty list.")
        for i, src in enumerate(node["sources"]):
            if "input_key" not in src or "from_rep" not in src:
                raise ValueError(f"mapping-json.{modality}.sources[{i}] must contain input_key and from_rep.")
            if "indices" in src:
                indices = src["indices"]
                if not isinstance(indices, list) or len(indices) == 0:
                    raise ValueError(
                        f"mapping-json.{modality}.sources[{i}].indices must be a non-empty list of integers."
                    )
                if any((not isinstance(j, int)) or isinstance(j, bool) for j in indices):
                    raise ValueError(
                        f"mapping-json.{modality}.sources[{i}].indices must contain integers only."
                    )
                if len(set(indices)) != len(indices):
                    raise ValueError(
                        f"mapping-json.{modality}.sources[{i}].indices must not contain duplicates."
                    )
    if mapping["state"]["output_key"] == mapping["action"]["output_key"]:
        raise ValueError("mapping-json state.output_key and action.output_key must be different.")


def _collect_data_files(root: Path) -> list[Path]:
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")
    files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}/chunk-*/file-*.parquet")
    return files


def _validate_metadata_inputs(dataset_root: Path, mapping: dict[str, Any]) -> None:
    info_path = dataset_root / "meta" / "info.json"
    modality_path = dataset_root / "meta" / "modality.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    if not modality_path.exists():
        raise FileNotFoundError(f"Missing modality.json: {modality_path}")

    info = _load_json(info_path)
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Invalid info.json features in {info_path}")

    state_output_key = mapping["state"]["output_key"]
    action_output_key = mapping["action"]["output_key"]
    if state_output_key in features or action_output_key in features:
        raise ValueError(
            "Output key already exists in input info.json. "
            f"state={state_output_key in features}, action={action_output_key in features}"
        )

    modality = _load_json(modality_path)
    if "state" not in modality or "action" not in modality:
        raise ValueError("modality.json must contain top-level 'state' and 'action'.")
    if "rot6d" in modality["state"] or "rot6d" in modality["action"]:
        raise ValueError("Input modality.json already has state.rot6d or action.rot6d.")


def _update_info_json(
    output_root: Path,
    mapping: dict[str, Any],
    state_dim: int,
    action_dim: int,
) -> None:
    info_path = output_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json: {info_path}")
    info = _load_json(info_path)
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"Invalid info.json features in {info_path}")

    state_output_key = mapping["state"]["output_key"]
    action_output_key = mapping["action"]["output_key"]
    if state_output_key in features or action_output_key in features:
        raise ValueError(
            f"Output key already exists in info.json features. "
            f"state={state_output_key in features}, action={action_output_key in features}"
        )

    features[state_output_key] = {"dtype": "float32", "shape": [state_dim], "names": None}
    features[action_output_key] = {"dtype": "float32", "shape": [action_dim], "names": None}
    info["features"] = features
    _save_json(info_path, info)


def _update_modality_json(
    output_root: Path,
    mapping: dict[str, Any],
    state_dim: int,
    action_dim: int,
) -> None:
    modality_path = output_root / "meta" / "modality.json"
    if not modality_path.exists():
        raise FileNotFoundError(f"Missing modality.json: {modality_path}")
    modality = _load_json(modality_path)

    if "state" not in modality or "action" not in modality:
        raise ValueError("modality.json must contain top-level 'state' and 'action'.")
    if "rot6d" in modality["state"] or "rot6d" in modality["action"]:
        raise ValueError("modality.json already contains state.rot6d or action.rot6d.")

    modality["state"]["rot6d"] = {
        "start": 0,
        "end": state_dim,
        "rotation_type": "rotation_6d",
        "absolute": True,
        "dtype": "float32",
        "original_key": mapping["state"]["output_key"],
    }
    modality["action"]["rot6d"] = {
        "start": 0,
        "end": action_dim,
        "rotation_type": "rotation_6d",
        "absolute": True,
        "dtype": "float32",
        "original_key": mapping["action"]["output_key"],
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

    # Pull only requested keys from the HF table to avoid decoding unrelated modalities (e.g. videos).
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


def _process_data_files(output_root: Path, mapping: dict[str, Any], dry_run: bool) -> tuple[int, int]:
    files = _collect_data_files(output_root)
    state_output_key = mapping["state"]["output_key"]
    action_output_key = mapping["action"]["output_key"]

    for file_path in files:
        df = pd.read_parquet(file_path)
        row_count = len(df)
        if row_count == 0:
            continue
        if state_output_key in df.columns or action_output_key in df.columns:
            raise ValueError(f"Output key already exists in file {file_path}")

        state_parts = [
            _convert_source_to_rot6d(df, src_cfg, row_count) for src_cfg in mapping["state"]["sources"]
        ]
        action_parts = [
            _convert_source_to_rot6d(df, src_cfg, row_count) for src_cfg in mapping["action"]["sources"]
        ]
        state_values = np.concatenate(state_parts, axis=1).astype(np.float32)
        action_values = np.concatenate(action_parts, axis=1).astype(np.float32)

        df[state_output_key] = [state_values[i] for i in range(row_count)]
        df[action_output_key] = [action_values[i] for i in range(row_count)]
        if not dry_run:
            df.to_parquet(file_path, index=False)

    return 6 * len(mapping["state"]["sources"]), 6 * len(mapping["action"]["sources"])


def _prepare_output_dir(dataset_root: Path, output_root: Path, overwrite_output: bool, dry_run: bool) -> None:
    if output_root.exists():
        if not overwrite_output:
            raise FileExistsError(f"Output dir already exists: {output_root}. Use --overwrite-output to replace it.")
        if not dry_run:
            shutil.rmtree(output_root)
    if not dry_run:
        shutil.copytree(dataset_root, output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add rotation_6d state/action features for a single LeRobot dataset, "
            "update info/modality metadata, and optionally recompute stats.json."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Input dataset root.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output dataset root.")
    parser.add_argument(
        "--mapping-json",
        type=Path,
        required=True,
        help="Mapping config JSON path. Each source supports optional indices (e.g. [3,4,5]) before conversion.",
    )
    parser.add_argument("--overwrite-output", action="store_true", help="Overwrite output-root if it exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and preview without writing.")
    parser.add_argument(
        "--recompute-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Recompute stats for new rot6d output keys only, then append/overwrite these keys in meta/stats.json "
            "(default: true)."
        ),
    )
    parser.add_argument(
        "--stats-output-name",
        type=str,
        default="stats.json",
        help="Stats filename under meta/. Current implementation supports only 'stats.json'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    mapping_path = args.mapping_json.resolve()

    if args.stats_output_name != "stats.json":
        raise ValueError("Only stats-output-name='stats.json' is supported.")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping json does not exist: {mapping_path}")

    mapping = _load_json(mapping_path)
    _validate_mapping(mapping)
    if output_root == dataset_root:
        raise ValueError("output-root must be different from dataset-root.")
    _validate_metadata_inputs(dataset_root, mapping)

    if args.dry_run:
        _collect_data_files(dataset_root)
        # Run conversion checks on first file only in dry-run.
        sample_file = _collect_data_files(dataset_root)[0]
        sample_df = pd.read_parquet(sample_file)
        row_count = len(sample_df)
        if row_count == 0:
            raise ValueError(f"Sample parquet has no rows: {sample_file}")
        _ = [_convert_source_to_rot6d(sample_df, src_cfg, row_count) for src_cfg in mapping["state"]["sources"]]
        _ = [_convert_source_to_rot6d(sample_df, src_cfg, row_count) for src_cfg in mapping["action"]["sources"]]
        print("[DRY-RUN] Rotation conversion validation succeeded on sample file:")
        print(f"  file={sample_file}")
        print(f"  state_output_key={mapping['state']['output_key']}")
        print(f"  action_output_key={mapping['action']['output_key']}")
        return

    _prepare_output_dir(dataset_root, output_root, args.overwrite_output, args.dry_run)
    state_dim, action_dim = _process_data_files(output_root, mapping, dry_run=False)
    _update_info_json(output_root, mapping, state_dim, action_dim)
    _update_modality_json(output_root, mapping, state_dim, action_dim)

    if args.recompute_stats:
        _recompute_stats(
            output_root,
            target_keys={mapping["state"]["output_key"], mapping["action"]["output_key"]},
            append_to_existing=True,
        )

    print("Completed rot6d augmentation.")
    print(f"  output_root: {output_root}")
    print(f"  state key: {mapping['state']['output_key']} dim={state_dim}")
    print(f"  action key: {mapping['action']['output_key']} dim={action_dim}")
    print(f"  stats: {output_root / 'meta' / args.stats_output_name}")


if __name__ == "__main__":
    main()
