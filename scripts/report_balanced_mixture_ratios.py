#!/usr/bin/env python3
"""Report dataset proportions for a named mixture after weight balancing."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Given a mixture key, compute and print per-dataset ratios after applying "
            "`balance_dataset_weights` logic."
        )
    )
    parser.add_argument(
        "--data-mix",
        type=str,
        required=True,
        help=f"Mixture key in DATASET_NAMED_MIXTURES. Available: {sorted(DATASET_NAMED_MIXTURES.keys())}",
    )
    parser.add_argument(
        "--config-yaml",
        type=str,
        default="starVLA/config/training/starvla_train_latent_world_vla_independent.yaml",
        help="Training config YAML used to read datasets.vla_data defaults.",
    )
    parser.add_argument(
        "--data-root-dir",
        type=str,
        default="/mnt/project_rlinf/jlchen/datasets",
        help="Dataset root directory. Defaults to /mnt/project_rlinf/jlchen/datasets.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default="/mnt/project_rlinf/jlchen/datasets/.hf_cache",
        help=(
            "HuggingFace cache root used by datasets/hub for this script. "
            "Avoids writing large cache files to system disk."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="Split mode used when measuring each dataset length.",
    )
    parser.add_argument(
        "--balance-dataset-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to apply `weight * dataset_length` before normalization.",
    )
    return parser


def _load_vla_data_cfg(config_yaml: Path) -> DictConfig:
    cfg = OmegaConf.load(str(config_yaml))

    if "datasets" in cfg and "vla_data" in cfg.datasets:
        base_cfg = cfg.datasets.vla_data
    elif "vla_data" in cfg:
        base_cfg = cfg.vla_data
    else:
        raise ValueError(
            f"Cannot find `datasets.vla_data` (or `vla_data`) in config: {config_yaml}"
        )

    return OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))


def _dedupe_mixture_entries(
    mixture_spec: list[tuple[str, float, str]],
) -> tuple[list[tuple[str, float, str]], int]:
    included: set[tuple[str, str]] = set()
    filtered: list[tuple[str, float, str]] = []
    duplicate_count = 0

    for d_name, d_weight, robot_type in mixture_spec:
        key = (d_name, robot_type)
        if key in included:
            duplicate_count += 1
            continue
        included.add(key)
        filtered.append((d_name, d_weight, robot_type))

    return filtered, duplicate_count


def _configure_hf_cache(cache_root: Path) -> None:
    cache_root = cache_root.expanduser()
    datasets_cache = cache_root / "datasets"
    hub_cache = cache_root / "hub"
    assets_cache = cache_root / "assets"
    tmp_cache = cache_root / "tmp"

    for path in (cache_root, datasets_cache, hub_cache, assets_cache, tmp_cache):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_ASSETS_CACHE"] = str(assets_cache)
    os.environ["TMPDIR"] = str(tmp_cache)


def _format_table(headers: list[str], rows: list[list[Any]]) -> str:
    str_rows = [[str(cell) for cell in row] for row in rows]
    col_widths = [
        max(len(headers[i]), *(len(row[i]) for row in str_rows)) for i in range(len(headers))
    ]

    def _fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * width for width in col_widths)
    lines = [_fmt(headers), sep]
    lines.extend(_fmt(row) for row in str_rows)
    return "\n".join(lines)


def main() -> None:
    args = _build_parser().parse_args()
    _configure_hf_cache(Path(args.hf_cache_dir))

    if args.data_mix not in DATASET_NAMED_MIXTURES:
        raise ValueError(
            f"Unknown data_mix `{args.data_mix}`. Available: {sorted(DATASET_NAMED_MIXTURES.keys())}"
        )

    config_yaml = Path(args.config_yaml)
    if not config_yaml.exists():
        raise FileNotFoundError(f"Config file not found: {config_yaml}")

    data_cfg = _load_vla_data_cfg(config_yaml)
    data_cfg.data_mix = args.data_mix

    if args.data_root_dir is not None:
        data_cfg.data_root_dir = args.data_root_dir

    if "data_root_dir" not in data_cfg:
        raise ValueError(
            "`data_root_dir` is missing. Provide it in config YAML or via --data-root-dir."
        )

    data_root_dir = Path(str(data_cfg.data_root_dir))
    if not data_root_dir.exists():
        raise FileNotFoundError(
            f"Data root dir does not exist: {data_root_dir}. Use --data-root-dir to override."
        )

    mixture_spec = DATASET_NAMED_MIXTURES[args.data_mix]
    filtered_spec, duplicate_count = _dedupe_mixture_entries(mixture_spec)

    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    mixture_dataset = get_vla_dataset(
        data_cfg=data_cfg,
        mode=args.mode,
        balance_dataset_weights=args.balance_dataset_weights,
    )
    empty_count = max(0, len(filtered_spec) - len(mixture_dataset.datasets))

    if len(mixture_dataset.datasets) == 0:
        raise ValueError("No valid datasets found in mixture after filtering empty datasets.")

    raw_weights = np.array(
        getattr(mixture_dataset, "_raw_dataset_sampling_weights", []), dtype=np.float64
    )
    effective_weights = np.array(
        getattr(mixture_dataset, "_effective_dataset_sampling_weights", []), dtype=np.float64
    )
    normalized_weights = mixture_dataset.dataset_sampling_weights.astype(np.float64)

    if raw_weights.size != len(mixture_dataset.datasets):
        raw_weights = np.full((len(mixture_dataset.datasets),), np.nan, dtype=np.float64)
    if effective_weights.size != len(mixture_dataset.datasets):
        effective_weights = np.full((len(mixture_dataset.datasets),), np.nan, dtype=np.float64)
    lengths = mixture_dataset.dataset_lengths.astype(np.int64)

    headers = [
        "Idx",
        "Dataset",
        "RawWeight",
        "Length",
        "EffectiveWeight",
        "Ratio(%)",
    ]
    table_rows: list[list[Any]] = []
    for idx, dataset in enumerate(mixture_dataset.datasets):
        table_rows.append(
            [
                idx,
                dataset.dataset_name,
                f"{raw_weights[idx]:.6g}",
                int(lengths[idx]),
                f"{effective_weights[idx]:.6g}",
                f"{normalized_weights[idx] * 100.0:.4f}",
            ]
        )

    print(
        "[INFO] "
        f"data_mix={args.data_mix} mode={args.mode} "
        f"hf_cache_dir={Path(args.hf_cache_dir)} "
        f"balance_dataset_weights={args.balance_dataset_weights} "
        f"datasets={len(mixture_dataset.datasets)} "
        f"duplicates_skipped={duplicate_count} empty_skipped={empty_count}"
    )
    print(_format_table(headers, table_rows))


if __name__ == "__main__":
    main()
