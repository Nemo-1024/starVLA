#!/usr/bin/env python3
"""Repair LeRobot episode offset metadata.

This script rewrites `meta/episodes/*/*.parquet` so that:
- `dataset_from_index` and `dataset_to_index` form a globally consistent range
  over all episodes;
- offsets are recomputed from `length` after sorting by `episode_index`.

Default mode is dry-run. Use `--apply` to write changes.
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
class EpisodeRef:
    path: Path
    row_idx: int
    episode_index: int
    length: int
    old_from: int
    old_to: int
    new_from: int = 0
    new_to: int = 0


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
        raise FileNotFoundError(f"no parquet files under: {episodes_dir}")
    return sorted(paths, key=_numeric_path_key)


def _read_refs(paths: list[Path]) -> list[EpisodeRef]:
    refs: list[EpisodeRef] = []
    needed_cols = ["episode_index", "length", "dataset_from_index", "dataset_to_index"]
    for path in paths:
        table = pq.read_table(path, columns=needed_cols)
        cols = table.column_names
        for col in needed_cols:
            if col not in cols:
                raise KeyError(f"missing `{col}` in {path}")

        ep = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
        length = np.asarray(table.column("length").to_pylist(), dtype=np.int64)
        old_from = np.asarray(table.column("dataset_from_index").to_pylist(), dtype=np.int64)
        old_to = np.asarray(table.column("dataset_to_index").to_pylist(), dtype=np.int64)

        if not (len(ep) == len(length) == len(old_from) == len(old_to)):
            raise ValueError(f"column length mismatch in {path}")

        for i in range(len(ep)):
            l = int(length[i])
            if l <= 0:
                raise ValueError(
                    f"invalid episode length in {path} row={i}: episode_index={int(ep[i])}, length={l}"
                )
            refs.append(
                EpisodeRef(
                    path=path,
                    row_idx=int(i),
                    episode_index=int(ep[i]),
                    length=l,
                    old_from=int(old_from[i]),
                    old_to=int(old_to[i]),
                )
            )
    return refs


def _plan_new_offsets(refs: list[EpisodeRef]) -> tuple[int, int]:
    if not refs:
        raise RuntimeError("no episode rows found")

    ep_ids = [r.episode_index for r in refs]
    if len(set(ep_ids)) != len(ep_ids):
        dup_count = len(ep_ids) - len(set(ep_ids))
        raise RuntimeError(
            "duplicate `episode_index` found in episodes metadata "
            f"(duplicates={dup_count}). "
            "Repair episode_index first, then repair offsets."
        )

    refs.sort(key=lambda r: r.episode_index)
    running = 0
    changed = 0
    for ref in refs:
        ref.new_from = int(running)
        ref.new_to = int(running + ref.length)
        running = ref.new_to
        if ref.new_from != ref.old_from or ref.new_to != ref.old_to:
            changed += 1
    return changed, running


def _apply(
    refs: list[EpisodeRef],
    *,
    dataset_root: Path,
    backup_dir: Path | None,
) -> None:
    refs_by_path: dict[Path, list[EpisodeRef]] = {}
    for ref in refs:
        refs_by_path.setdefault(ref.path, []).append(ref)

    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(refs_by_path.keys(), key=_numeric_path_key)
    for i, path in enumerate(paths):
        table = pq.read_table(path)
        row_refs = refs_by_path[path]

        from_idx = table.schema.get_field_index("dataset_from_index")
        to_idx = table.schema.get_field_index("dataset_to_index")
        if from_idx < 0 or to_idx < 0:
            raise KeyError(f"dataset_from_index/dataset_to_index not found in {path}")

        from_vals = np.asarray(table.column("dataset_from_index").to_pylist(), dtype=np.int64)
        to_vals = np.asarray(table.column("dataset_to_index").to_pylist(), dtype=np.int64)
        for ref in row_refs:
            from_vals[ref.row_idx] = ref.new_from
            to_vals[ref.row_idx] = ref.new_to

        from_type = table.schema.field(from_idx).type
        to_type = table.schema.field(to_idx).type
        table = table.set_column(from_idx, "dataset_from_index", pa.array(from_vals.tolist(), type=from_type))
        table = table.set_column(to_idx, "dataset_to_index", pa.array(to_vals.tolist(), type=to_type))

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, tmp_path, compression="snappy")

        if backup_dir is not None:
            rel = path.relative_to(dataset_root)
            backup_path = backup_dir / rel
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_path)

        tmp_path.replace(path)
        if (i + 1) % 100 == 0 or i == len(paths) - 1:
            print(f"[apply] rewritten {i + 1}/{len(paths)} files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair LeRobot episode dataset_from/to offsets.")
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
        help="Optional backup directory for original episode parquet files.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=8,
        help="Preview first N episode rows after sorting by episode_index.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    paths = _episode_paths(dataset_root)
    refs = _read_refs(paths)
    changed_rows, total_steps = _plan_new_offsets(refs)

    print(f"[scan] dataset_root={dataset_root}")
    print(f"[scan] episode_files={len(paths)} episode_rows={len(refs)} total_steps={total_steps}")
    print(f"[scan] rows_to_change={changed_rows}")

    preview_n = max(0, int(args.preview_rows))
    if preview_n > 0:
        print("[preview] first rows by episode_index:")
        for ref in refs[:preview_n]:
            print(
                f"  ep={ref.episode_index} len={ref.length} "
                f"old=[{ref.old_from},{ref.old_to}) new=[{ref.new_from},{ref.new_to}) "
                f"file={ref.path.name} row={ref.row_idx}"
            )

    if not args.apply:
        print("[dry-run] no files modified. Re-run with --apply to write changes.")
        return

    print("[apply] rewriting episode offset columns ...")
    _apply(refs, dataset_root=dataset_root, backup_dir=args.backup_dir)
    print("[done] repair completed.")


if __name__ == "__main__":
    main()

