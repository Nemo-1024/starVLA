#!/usr/bin/env python3
"""
In-place conversion for epic_kitchens_100_lerobot:
- Replace state/action columns in all episode parquet files
- Use:
    - state.dummy_state  (14-dim)
    - action.dummy_action (14-dim)
- Update meta/info.json, meta/modality.json, meta/stats_gr00t.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_DATASET_DIR = Path("/mnt/project_rlinf/jlchen/datasets/epic_kitchens_100_lerobot")
DEFAULT_STATE_KEY = "state.dummy_state"
DEFAULT_ACTION_KEY = "action.dummy_action"
DEFAULT_STATE_MODALITY_NAME = "dummy_state"
DEFAULT_ACTION_MODALITY_NAME = "dummy_action"
DEFAULT_DIM = 14


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="In-place convert LeRobot dataset to dummy 14-dim state/action.")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Target dataset directory (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--state_key",
        type=str,
        default=DEFAULT_STATE_KEY,
        help=f"New state key in parquet/info/stats (default: {DEFAULT_STATE_KEY})",
    )
    parser.add_argument(
        "--action_key",
        type=str,
        default=DEFAULT_ACTION_KEY,
        help=f"New action key in parquet/info/stats (default: {DEFAULT_ACTION_KEY})",
    )
    parser.add_argument(
        "--state_modality_name",
        type=str,
        default=DEFAULT_STATE_MODALITY_NAME,
        help=f"State modality name in modality.json (default: {DEFAULT_STATE_MODALITY_NAME})",
    )
    parser.add_argument(
        "--action_modality_name",
        type=str,
        default=DEFAULT_ACTION_MODALITY_NAME,
        help=f"Action modality name in modality.json (default: {DEFAULT_ACTION_MODALITY_NAME})",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=DEFAULT_DIM,
        help=f"State/action vector dim (default: {DEFAULT_DIM})",
    )
    parser.add_argument(
        "--chunk_glob",
        type=str,
        default="chunk-*",
        help="Chunk directory glob under data/, e.g. chunk-000 or chunk-*",
    )
    return parser.parse_args()


def _zero_vec_stats(dim: int) -> dict:
    zero = [0.0] * dim
    return {
        "mean": zero,
        "std": zero,
        "min": zero,
        "max": zero,
        "q01": zero,
        "q99": zero,
    }


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict, indent: int = 2) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent)


def _collect_episode_files(dataset_dir: Path, chunk_glob: str) -> list[Path]:
    data_dir = dataset_dir / "data"
    paths: list[Path] = []
    for chunk_dir in sorted(data_dir.glob(chunk_glob)):
        if not chunk_dir.is_dir():
            continue
        paths.extend(sorted(chunk_dir.glob("episode_*.parquet")))
    return paths


def _replace_parquet_columns(
    parquet_path: Path,
    state_key: str,
    action_key: str,
    dim: int,
) -> int:
    df = pd.read_parquet(parquet_path)
    frame_count = len(df)

    state_cols = [col for col in df.columns if col.startswith("state.")]
    action_cols = [col for col in df.columns if col.startswith("action.")]
    drop_cols = [col for col in (state_cols + action_cols) if col not in {state_key, action_key}]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    state_values = np.zeros((frame_count, dim), dtype=np.float32)
    action_values = np.zeros((frame_count, dim), dtype=np.float32)
    df[state_key] = list(state_values)
    df[action_key] = list(action_values)
    df.to_parquet(parquet_path, index=False)
    return frame_count


def _update_info_json(
    dataset_dir: Path,
    state_key: str,
    action_key: str,
    dim: int,
) -> None:
    info_path = dataset_dir / "meta" / "info.json"
    info = _load_json(info_path)
    features = info.setdefault("features", {})
    remove_keys = [k for k in features if isinstance(k, str) and (k.startswith("state.") or k.startswith("action."))]
    for key in remove_keys:
        features.pop(key, None)
    features[state_key] = {"dtype": "float32", "shape": [dim]}
    features[action_key] = {"dtype": "float32", "shape": [dim]}
    _save_json(info_path, info, indent=2)


def _update_modality_json(
    dataset_dir: Path,
    state_key: str,
    action_key: str,
    state_modality_name: str,
    action_modality_name: str,
    dim: int,
) -> None:
    modality_path = dataset_dir / "meta" / "modality.json"
    modality = _load_json(modality_path)
    modality["state"] = {
        state_modality_name: {
            "start": 0,
            "end": dim,
            "absolute": True,
            "dtype": "float32",
            "original_key": state_key,
        }
    }
    modality["action"] = {
        action_modality_name: {
            "start": 0,
            "end": dim,
            "absolute": True,
            "dtype": "float32",
            "original_key": action_key,
        }
    }
    _save_json(modality_path, modality, indent=2)


def _update_stats_json(
    dataset_dir: Path,
    state_key: str,
    action_key: str,
    dim: int,
) -> None:
    stats_path = dataset_dir / "meta" / "stats_gr00t.json"
    stats = _load_json(stats_path)
    remove_keys = [k for k in stats if isinstance(k, str) and (k.startswith("state.") or k.startswith("action."))]
    for key in remove_keys:
        stats.pop(key, None)
    stats[state_key] = _zero_vec_stats(dim)
    stats[action_key] = _zero_vec_stats(dim)
    _save_json(stats_path, stats, indent=4)


def main() -> None:
    args = _parse_args()
    if args.dim <= 0:
        raise ValueError(f"--dim must be > 0, got {args.dim}")
    if not args.state_key.startswith("state."):
        raise ValueError(f"--state_key must start with 'state.', got {args.state_key}")
    if not args.action_key.startswith("action."):
        raise ValueError(f"--action_key must start with 'action.', got {args.action_key}")
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {args.dataset_dir}")

    parquet_files = _collect_episode_files(args.dataset_dir, args.chunk_glob)
    if not parquet_files:
        raise FileNotFoundError(
            f"No episode parquet files found under {args.dataset_dir / 'data'} with chunk glob {args.chunk_glob}"
        )

    total_frames = 0
    for parquet_path in tqdm(parquet_files, desc="Converting parquet (in-place)", unit="file"):
        total_frames += _replace_parquet_columns(
            parquet_path=parquet_path,
            state_key=args.state_key,
            action_key=args.action_key,
            dim=int(args.dim),
        )

    _update_info_json(
        dataset_dir=args.dataset_dir,
        state_key=args.state_key,
        action_key=args.action_key,
        dim=int(args.dim),
    )
    _update_modality_json(
        dataset_dir=args.dataset_dir,
        state_key=args.state_key,
        action_key=args.action_key,
        state_modality_name=args.state_modality_name,
        action_modality_name=args.action_modality_name,
        dim=int(args.dim),
    )
    _update_stats_json(
        dataset_dir=args.dataset_dir,
        state_key=args.state_key,
        action_key=args.action_key,
        dim=int(args.dim),
    )

    print(f"Converted dataset in place: {args.dataset_dir}")
    print(f"Parquet files updated: {len(parquet_files)}")
    print(f"Total frames touched: {total_frames}")
    print(f"State key: {args.state_key}, dim={args.dim}")
    print(f"Action key: {args.action_key}, dim={args.dim}")


if __name__ == "__main__":
    main()
