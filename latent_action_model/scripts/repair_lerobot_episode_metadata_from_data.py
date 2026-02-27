#!/usr/bin/env python3
"""Repair LeRobot episode metadata from data parquet ground truth.

Rebuilds the following columns in `meta/episodes/*/*.parquet` using
`data/*/*.parquet` `episode_index` sequence as source of truth:

- `length`
- `dataset_from_index`
- `dataset_to_index`

Default mode is dry-run. Use `--apply` to write files in-place.
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
    old_length: int
    old_from: int
    old_to: int


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
    paths = sorted(episodes_dir.glob("*/*.parquet"), key=_numeric_path_key)
    if not paths:
        raise FileNotFoundError(f"no episode parquet files under: {episodes_dir}")
    return paths


def _data_paths(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir not found: {data_dir}")
    paths = sorted(data_dir.glob("*/*.parquet"), key=_numeric_path_key)
    if not paths:
        raise FileNotFoundError(f"no data parquet files under: {data_dir}")
    return paths


def _load_episode_refs(paths: list[Path]) -> list[EpisodeRef]:
    refs: list[EpisodeRef] = []
    for path in paths:
        table = pq.read_table(
            path,
            columns=["episode_index", "length", "dataset_from_index", "dataset_to_index"],
        )
        ep = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
        length = np.asarray(table.column("length").to_pylist(), dtype=np.int64)
        from_idx = np.asarray(table.column("dataset_from_index").to_pylist(), dtype=np.int64)
        to_idx = np.asarray(table.column("dataset_to_index").to_pylist(), dtype=np.int64)
        if not (len(ep) == len(length) == len(from_idx) == len(to_idx)):
            raise ValueError(f"column length mismatch in {path}")
        for i in range(len(ep)):
            refs.append(
                EpisodeRef(
                    path=path,
                    row_idx=int(i),
                    episode_index=int(ep[i]),
                    old_length=int(length[i]),
                    old_from=int(from_idx[i]),
                    old_to=int(to_idx[i]),
                )
            )
    return refs


def _scan_data_episode_layout(data_paths: list[Path]) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """Return (counts, first_abs, last_abs) by episode_index."""
    counts: dict[int, int] = {}
    first_abs: dict[int, int] = {}
    last_abs: dict[int, int] = {}
    offset = 0

    for i, path in enumerate(data_paths):
        table = pq.read_table(path, columns=["episode_index"])
        ep = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
        n = int(ep.size)
        if n == 0:
            continue

        uniq, cnt = np.unique(ep, return_counts=True)
        for u, c in zip(uniq.tolist(), cnt.tolist()):
            eid = int(u)
            counts[eid] = counts.get(eid, 0) + int(c)

        uniq_first, first_local = np.unique(ep, return_index=True)
        for u, idx0 in zip(uniq_first.tolist(), first_local.tolist()):
            eid = int(u)
            abs_idx = offset + int(idx0)
            if eid not in first_abs:
                first_abs[eid] = abs_idx
            else:
                first_abs[eid] = min(first_abs[eid], abs_idx)

        rev = ep[::-1]
        uniq_last, last_rev_local = np.unique(rev, return_index=True)
        for u, rev_idx in zip(uniq_last.tolist(), last_rev_local.tolist()):
            eid = int(u)
            idx1 = n - 1 - int(rev_idx)
            abs_idx = offset + idx1
            if eid not in last_abs:
                last_abs[eid] = abs_idx
            else:
                last_abs[eid] = max(last_abs[eid], abs_idx)

        offset += n
        if (i + 1) % 256 == 0 or i == len(data_paths) - 1:
            print(f"[scan-data] processed {i + 1}/{len(data_paths)} files, rows={offset}")

    return counts, first_abs, last_abs


def _build_new_meta(
    refs: list[EpisodeRef],
    counts: dict[int, int],
    first_abs: dict[int, int],
    last_abs: dict[int, int],
) -> tuple[dict[int, int], dict[int, int], dict[int, int], int]:
    ep_ids = [r.episode_index for r in refs]
    if len(set(ep_ids)) != len(ep_ids):
        dup = len(ep_ids) - len(set(ep_ids))
        raise RuntimeError(
            f"duplicate episode_index in episodes metadata (duplicates={dup}); "
            "repair episode_index first."
        )

    meta_set = set(ep_ids)
    data_set = set(counts.keys())
    missing_in_data = sorted(meta_set - data_set)
    extra_in_data = sorted(data_set - meta_set)
    if missing_in_data or extra_in_data:
        raise RuntimeError(
            "episode index set mismatch between meta and data: "
            f"missing_in_data={len(missing_in_data)}, extra_in_data={len(extra_in_data)}"
        )

    new_len: dict[int, int] = {}
    new_from: dict[int, int] = {}
    new_to: dict[int, int] = {}
    non_contiguous = 0
    for eid in sorted(meta_set):
        c = int(counts[eid])
        f = int(first_abs[eid])
        l = int(last_abs[eid])
        if c != (l - f + 1):
            non_contiguous += 1
        new_len[eid] = c
        new_from[eid] = f
        new_to[eid] = f + c

    return new_len, new_from, new_to, non_contiguous


def _apply_updates(
    refs: list[EpisodeRef],
    *,
    dataset_root: Path,
    backup_dir: Path | None,
    new_len: dict[int, int],
    new_from: dict[int, int],
    new_to: dict[int, int],
) -> None:
    refs_by_path: dict[Path, list[EpisodeRef]] = {}
    for ref in refs:
        refs_by_path.setdefault(ref.path, []).append(ref)

    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(refs_by_path.keys(), key=_numeric_path_key)
    for i, path in enumerate(paths):
        table = pq.read_table(path)
        ep_vals = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
        len_vals = np.asarray(table.column("length").to_pylist(), dtype=np.int64)
        from_vals = np.asarray(table.column("dataset_from_index").to_pylist(), dtype=np.int64)
        to_vals = np.asarray(table.column("dataset_to_index").to_pylist(), dtype=np.int64)

        row_refs = refs_by_path[path]
        for ref in row_refs:
            eid = ref.episode_index
            len_vals[ref.row_idx] = new_len[eid]
            from_vals[ref.row_idx] = new_from[eid]
            to_vals[ref.row_idx] = new_to[eid]

        len_idx = table.schema.get_field_index("length")
        from_idx = table.schema.get_field_index("dataset_from_index")
        to_idx = table.schema.get_field_index("dataset_to_index")
        if len_idx < 0 or from_idx < 0 or to_idx < 0:
            raise KeyError(f"required columns missing in {path}")

        table = table.set_column(
            len_idx,
            "length",
            pa.array(len_vals.tolist(), type=table.schema.field(len_idx).type),
        )
        table = table.set_column(
            from_idx,
            "dataset_from_index",
            pa.array(from_vals.tolist(), type=table.schema.field(from_idx).type),
        )
        table = table.set_column(
            to_idx,
            "dataset_to_index",
            pa.array(to_vals.tolist(), type=table.schema.field(to_idx).type),
        )

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, tmp_path, compression="snappy")

        if backup_dir is not None:
            rel = path.relative_to(dataset_root)
            backup_path = backup_dir / rel
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_path)

        tmp_path.replace(path)
        if (i + 1) % 100 == 0 or i == len(paths) - 1:
            print(f"[apply] rewritten {i + 1}/{len(paths)} episode files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair LeRobot episode metadata from data parquet.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root, e.g. /mnt/project_rlinf/jlchen/datasets/droid_1.0.1",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite episode parquet files in-place. Default is dry-run.",
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
        help="Print first N episodes (sorted by episode_index) before/after preview.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    episode_paths = _episode_paths(dataset_root)
    data_paths = _data_paths(dataset_root)
    refs = _load_episode_refs(episode_paths)
    counts, first_abs, last_abs = _scan_data_episode_layout(data_paths)
    new_len, new_from, new_to, non_contiguous = _build_new_meta(refs, counts, first_abs, last_abs)

    changed_len = 0
    changed_from_to = 0
    for ref in refs:
        eid = ref.episode_index
        if ref.old_length != new_len[eid]:
            changed_len += 1
        if ref.old_from != new_from[eid] or ref.old_to != new_to[eid]:
            changed_from_to += 1

    print(f"[scan] dataset_root={dataset_root}")
    print(f"[scan] episode_files={len(episode_paths)} episode_rows={len(refs)}")
    print(f"[scan] changed_length_rows={changed_len} changed_from_to_rows={changed_from_to}")
    print(f"[scan] non_contiguous_episodes_in_data={non_contiguous}")

    preview_n = max(0, int(args.preview_rows))
    if preview_n > 0:
        print("[preview] first episodes by episode_index:")
        for eid in sorted(new_len.keys())[:preview_n]:
            ref = next(r for r in refs if r.episode_index == eid)
            print(
                f"  ep={eid} old_len={ref.old_length} new_len={new_len[eid]} "
                f"old=[{ref.old_from},{ref.old_to}) new=[{new_from[eid]},{new_to[eid]})"
            )

    if non_contiguous > 0:
        raise RuntimeError(
            "Detected non-contiguous episode spans in data files. "
            "Aborting automatic rewrite to avoid corrupt metadata."
        )

    if not args.apply:
        print("[dry-run] no files modified. Re-run with --apply to write changes.")
        return

    print("[apply] rewriting episode metadata ...")
    _apply_updates(
        refs,
        dataset_root=dataset_root,
        backup_dir=args.backup_dir,
        new_len=new_len,
        new_from=new_from,
        new_to=new_to,
    )
    print("[done] repair completed.")


if __name__ == "__main__":
    main()

