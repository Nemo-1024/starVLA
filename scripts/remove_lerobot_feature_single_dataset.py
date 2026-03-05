#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path


def _ensure_lerobot_on_path() -> None:
    if importlib.util.find_spec("lerobot") is not None:
        return

    candidates = [
        Path.cwd() / ".." / "lerobot" / "src",
        Path(__file__).resolve().parents[2] / "lerobot" / "src",
        Path("/mnt/project_rlinf/jlchen/code/lerobot/src"),
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            if importlib.util.find_spec("lerobot") is not None:
                return

    raise ModuleNotFoundError(
        "Cannot import 'lerobot'. Please install lerobot or add its src path to PYTHONPATH."
    )


_ensure_lerobot_on_path()

from lerobot.datasets.dataset_tools import remove_feature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove one or more feature keys from a local LeRobot dataset."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Local LeRobot dataset root path (contains data/ and meta/).",
    )
    parser.add_argument(
        "--remove-key",
        type=str,
        action="append",
        required=True,
        help="Feature key to remove. Repeat this flag for multiple keys, e.g. --remove-key actions.joint_position.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output dataset path. Default: <dataset-path>_feature_removed. Ignored when --inplace is set.",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        default=None,
        help="repo_id metadata for output dataset. Default: output directory name.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite output path if it already exists. For --inplace mode, stale temp dir is always cleaned.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help=(
            "Overwrite dataset-path in place. Internally writes to a temp directory first, then atomically swaps "
            "into the original path."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned changes without writing output dataset.",
    )
    return parser.parse_args()


def _normalize_remove_keys(raw_keys: list[str]) -> list[str]:
    out: list[str] = []
    for raw in raw_keys:
        parts = [p.strip() for p in raw.split(",")]
        for part in parts:
            if part:
                out.append(part)
    deduped: list[str] = []
    seen = set()
    for key in out:
        if key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def _replace_dataset_inplace(dataset_path: Path, temp_output_path: Path, backup_path: Path) -> None:
    if not temp_output_path.exists():
        raise FileNotFoundError(f"Temporary output dataset missing: {temp_output_path}")
    if backup_path.exists():
        raise FileExistsError(
            f"Backup path already exists: {backup_path}. Please remove it or choose non-inplace mode."
        )

    dataset_path.rename(backup_path)
    try:
        temp_output_path.rename(dataset_path)
    except Exception as exc:
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        backup_path.rename(dataset_path)
        raise RuntimeError("Failed to swap temp dataset into original path. Rolled back to original dataset.") from exc

    shutil.rmtree(backup_path)


def main() -> None:
    args = _parse_args()

    dataset_path = args.dataset_path.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not (dataset_path / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Missing meta/info.json under dataset path: {dataset_path}")

    remove_keys = _normalize_remove_keys(args.remove_key)
    if not remove_keys:
        raise ValueError("No valid remove key is provided.")
    if args.inplace and args.output_path is not None:
        raise ValueError("--output-path cannot be used with --inplace.")

    repo_id = dataset_path.name
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_path)

    feature_keys = set(meta.features.keys())
    missing = [k for k in remove_keys if k not in feature_keys]
    if missing:
        preview = ", ".join(sorted(feature_keys))
        raise KeyError(
            "Some remove keys are not present in dataset features.\n"
            f"Missing: {missing}\n"
            f"Available features: {preview}"
        )

    if args.inplace:
        output_path = dataset_path.parent / f".{dataset_path.name}.tmp_feature_removed"
        backup_path = dataset_path.parent / f".{dataset_path.name}.backup_before_feature_remove"
    else:
        backup_path = None
        if args.output_path is None:
            output_path = dataset_path.parent / f"{dataset_path.name}_feature_removed"
        else:
            output_path = args.output_path.resolve()

        if output_path == dataset_path:
            raise ValueError(
                "output-path cannot be the same as dataset-path. "
                "Please use --inplace for overwrite behavior."
            )

    print("Plan:")
    print(f"  dataset_path: {dataset_path}")
    print(f"  repo_id: {repo_id}")
    print(f"  total_episodes: {meta.total_episodes}")
    print(f"  total_frames: {meta.total_frames}")
    print(f"  remove_keys: {remove_keys}")
    print(f"  inplace: {args.inplace}")
    print(f"  output_path: {output_path}")
    if args.inplace and backup_path is not None:
        print(f"  backup_path: {backup_path}")

    if args.dry_run:
        print("Dry run finished. No files written.")
        return

    if output_path.exists():
        if args.inplace:
            shutil.rmtree(output_path)
        else:
            if not args.overwrite_output:
                raise FileExistsError(
                    f"Output path already exists: {output_path}. Use --overwrite-output to replace it."
                )
            shutil.rmtree(output_path)

    output_repo_id = (args.output_repo_id or output_path.name) if not args.inplace else repo_id

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_path,
        download_videos=False,
    )

    new_dataset = remove_feature(
        dataset=dataset,
        feature_names=remove_keys,
        output_dir=output_path,
        repo_id=output_repo_id,
    )
    remaining_feature_count = len(new_dataset.meta.features)

    if args.inplace:
        if backup_path is None:
            raise RuntimeError("Internal error: backup path is not initialized for inplace mode.")
        _replace_dataset_inplace(dataset_path=dataset_path, temp_output_path=output_path, backup_path=backup_path)
        final_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_path)
        final_root = dataset_path
        final_repo_id = repo_id
        remaining_feature_count = len(final_meta.features)
    else:
        final_root = new_dataset.root
        final_repo_id = new_dataset.repo_id

    print("Done.")
    print(f"  output_dataset_root: {final_root}")
    print(f"  output_repo_id: {final_repo_id}")
    print(f"  removed_keys: {remove_keys}")
    print(f"  remaining_feature_count: {remaining_feature_count}")


if __name__ == "__main__":
    main()
