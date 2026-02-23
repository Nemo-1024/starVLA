#!/usr/bin/env python3
"""Repair LeRobot episode metadata when episode_index resets per shard/file.

This script rewrites `meta/episodes/*/*.parquet` in-place (or dry-run) so that
`episode_index` becomes globally unique and aligned with global data indexing.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class FilePlan:
    path: Path
    num_rows: int
    local_min: int
    local_max: int
    base_offset: int
    new_min: int
    new_max: int


def _numeric_path_key(path: Path) -> tuple[int, int]:
    chunk_match = re.search(r"(\d+)", path.parent.name)
    file_match = re.search(r"(\d+)", path.stem)
    chunk_idx = int(chunk_match.group(1)) if chunk_match else 0
    file_idx = int(file_match.group(1)) if file_match else 0
    return chunk_idx, file_idx


def _episode_paths(dataset_root: Path) -> list[Path]:
    episodes_dir = dataset_root / "meta" / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"episodes dir not found: {episodes_dir}")
    paths = list(episodes_dir.glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet under: {episodes_dir}")
    return sorted(paths, key=_numeric_path_key)


def _read_episode_index_col(path: Path) -> np.ndarray:
    table = pq.read_table(path, columns=["episode_index"])
    arr = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"episode_index is not 1D in {path}")
    return arr


def _build_plan(paths: list[Path]) -> tuple[list[FilePlan], int, int]:
    plans: list[FilePlan] = []
    running_total = 0
    raw_unique: set[int] = set()
    total_rows = 0

    for path in paths:
        local_ids = _read_episode_index_col(path)
        if local_ids.size == 0:
            continue
        local_min = int(local_ids.min())
        local_max = int(local_ids.max())

        # Shift local ids to global episode id space while preserving local values.
        # Typical case: local ids start from 0 within each file.
        base_offset = running_total - local_min
        new_ids = local_ids + base_offset

        plans.append(
            FilePlan(
                path=path,
                num_rows=int(local_ids.size),
                local_min=local_min,
                local_max=local_max,
                base_offset=base_offset,
                new_min=int(new_ids.min()),
                new_max=int(new_ids.max()),
            )
        )

        total_rows += int(local_ids.size)
        running_total += int(local_ids.size)
        raw_unique.update(int(x) for x in local_ids.tolist())

    return plans, total_rows, len(raw_unique)


def _apply_plan(
    plans: list[FilePlan],
    dataset_root: Path,
    backup_dir: Path | None,
) -> None:
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)

    for i, plan in enumerate(plans):
        table = pq.read_table(plan.path)
        if "episode_index" not in table.column_names:
            raise KeyError(f"episode_index not found in {plan.path}")
        old_ids = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
        new_ids = old_ids + plan.base_offset

        field_idx = table.schema.get_field_index("episode_index")
        target_type = table.schema.field(field_idx).type
        new_col = pa.array(new_ids.tolist(), type=target_type)
        new_table = table.set_column(field_idx, "episode_index", new_col)

        tmp_path = plan.path.with_suffix(plan.path.suffix + ".tmp")
        pq.write_table(new_table, tmp_path, compression="snappy")

        if backup_dir is not None:
            rel_path = plan.path.relative_to(dataset_root)
            backup_path = backup_dir / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(plan.path, backup_path)

        tmp_path.replace(plan.path)
        if (i + 1) % 100 == 0 or i == len(plans) - 1:
            print(f"[apply] rewritten {i + 1}/{len(plans)} files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair LeRobot episodes episode_index to global ids.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root, e.g. /mnt/project_rlinf/jlchen/datasets/droid_1.0.1",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite parquet files in-place. Default is dry-run.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Optional backup directory for original parquet files.",
    )
    parser.add_argument(
        "--preview-files",
        type=int,
        default=5,
        help="Number of files to preview in plan output.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    paths = _episode_paths(dataset_root)
    plans, total_rows, raw_unique_count = _build_plan(paths)

    if not plans:
        raise RuntimeError("No non-empty episode parquet files found.")

    print(f"[scan] dataset_root={dataset_root}")
    print(f"[scan] episode_files={len(paths)} episode_rows={total_rows}")
    print(f"[scan] raw_episode_index_unique={raw_unique_count} duplicates={total_rows - raw_unique_count}")
    print(
        f"[scan] planned_global_range=[{plans[0].new_min}, {plans[-1].new_max}] "
        f"(may contain gaps if local ids are non-contiguous)"
    )

    preview_n = max(0, int(args.preview_files))
    if preview_n > 0:
        print("[preview] first files:")
        for plan in plans[:preview_n]:
            print(
                f"  {plan.path} rows={plan.num_rows} "
                f"local=[{plan.local_min},{plan.local_max}] "
                f"base_offset={plan.base_offset} new=[{plan.new_min},{plan.new_max}]"
            )

    if not args.apply:
        print("[dry-run] no file modified. Re-run with --apply to rewrite episode_index.")
        return

    print("[apply] rewriting episode parquet files ...")
    _apply_plan(plans, dataset_root=dataset_root, backup_dir=args.backup_dir)
    print("[done] repair completed.")


if __name__ == "__main__":
    main()

