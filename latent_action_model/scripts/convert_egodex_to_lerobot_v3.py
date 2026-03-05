#!/usr/bin/env python3
"""
Batch-convert EgoDex root directory into one integrated LeRobot v3.0 dataset.

This script reuses `convert_egodex_task_to_lerobot_v3.py` by first staging all
task samples into one virtual task directory:

  <stage_dir>/combined_task/
    actions/{task}__{sample}.pt
    qpos/{task}__{sample}.pt
    videos/{task}__{sample}.mp4
    metas/{task}__{sample}.txt

Then it runs the single-task converter once on that staged directory, producing
a unified output dataset.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-convert EgoDex root dir into one integrated LeRobot v3 dataset."
    )
    parser.add_argument("--input-root-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional working directory for staged links/files. Defaults to <output-dir>.parent.",
    )
    parser.add_argument(
        "--stage-name",
        type=str,
        default=None,
        help=(
            "Name of staging directory under --work-dir. "
            "Default is auto-generated as .tmp_egodex_stage_<pid>."
        ),
    )
    parser.add_argument(
        "--link-mode",
        type=str,
        default="symlink",
        choices=["symlink", "hardlink", "copy"],
        help="How to stage files before conversion.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-stage", action="store_true")

    # Forwarded options for convert_egodex_task_to_lerobot_v3.py
    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument("--resize-height", type=int, default=360)
    parser.add_argument("--video-codec", type=str, default="libx264")
    parser.add_argument("--video-pix-fmt", type=str, default="yuv420p")
    parser.add_argument("--gop", type=int, default=12)
    parser.add_argument("--chunk-size-episodes", type=int, default=1000)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--strict-validate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--stage-progress-every",
        type=int,
        default=2000,
        help="Print one staging progress line every N samples. Set <=0 to disable.",
    )
    return parser.parse_args()


def _stems_for_ext(directory: Path, ext: str) -> set[str]:
    return {p.stem for p in directory.glob(f"*{ext}") if p.is_file()}


def _sample_sort_key(sample_id: str) -> tuple:
    # Deterministic natural-ish ordering: split by digit runs.
    tokens: list[object] = []
    cur = ""
    in_digit = False
    for ch in sample_id:
        if ch.isdigit():
            if not in_digit and cur:
                tokens.append(cur)
                cur = ""
            in_digit = True
            cur += ch
        else:
            if in_digit and cur:
                tokens.append(int(cur))
                cur = ""
            in_digit = False
            cur += ch
    if cur:
        tokens.append(int(cur) if in_digit else cur)
    return tuple(tokens)


def _safe_rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


def _is_task_dir(path: Path) -> bool:
    required = [path / "actions", path / "qpos", path / "videos", path / "metas"]
    return path.is_dir() and all(p.is_dir() for p in required)


def _discover_task_dirs(input_root_dir: Path) -> list[Path]:
    if not input_root_dir.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root_dir}")
    if not input_root_dir.is_dir():
        raise NotADirectoryError(f"--input-root-dir must be a directory: {input_root_dir}")

    # Allow using a single task directory directly for quick smoke tests.
    if _is_task_dir(input_root_dir):
        return [input_root_dir]

    task_dirs: list[Path] = []
    for child in sorted(input_root_dir.iterdir(), key=lambda p: p.name):
        if _is_task_dir(child):
            task_dirs.append(child)
    if not task_dirs:
        raise RuntimeError(
            "No valid task directories found (expecting actions/qpos/videos/metas under each task dir)."
        )
    return task_dirs


def _discover_sample_ids(task_dir: Path) -> list[str]:
    stems_actions = _stems_for_ext(task_dir / "actions", ".pt")
    stems_qpos = _stems_for_ext(task_dir / "qpos", ".pt")
    stems_videos = _stems_for_ext(task_dir / "videos", ".mp4")
    stems_metas = _stems_for_ext(task_dir / "metas", ".txt")
    sample_ids = stems_actions & stems_qpos & stems_videos & stems_metas
    return sorted(sample_ids, key=_sample_sort_key)


def _safe_name(text: str) -> str:
    return "".join(c if (c.isalnum() or c in {"_", "-"}) else "_" for c in text)


def _link_file(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"Destination already exists while staging: {dst}")
    if link_mode == "symlink":
        # Use absolute path symlink for robustness when working dir changes.
        dst.symlink_to(src.resolve())
    elif link_mode == "hardlink":
        os.link(src, dst)
    elif link_mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {link_mode}")


def _stage_combined_task(
    input_root_dir: Path,
    stage_combined_task_dir: Path,
    link_mode: str,
    max_episodes: int | None,
    stage_progress_every: int,
) -> tuple[int, int]:
    task_dirs = _discover_task_dirs(input_root_dir)
    actions_dir = stage_combined_task_dir / "actions"
    qpos_dir = stage_combined_task_dir / "qpos"
    videos_dir = stage_combined_task_dir / "videos"
    metas_dir = stage_combined_task_dir / "metas"
    for d in [actions_dir, qpos_dir, videos_dir, metas_dir]:
        d.mkdir(parents=True, exist_ok=True)

    total_tasks = 0
    total_samples = 0
    seen_staged_ids: set[str] = set()

    for task_dir in task_dirs:
        task_name = _safe_name(task_dir.name)
        sample_ids = _discover_sample_ids(task_dir)
        if not sample_ids:
            print(
                f"[WARN] Skip task with no valid intersecting samples: {task_dir.name}",
                flush=True,
            )
            continue
        total_tasks += 1
        print(f"[INFO] Staging task `{task_dir.name}`: {len(sample_ids)} samples", flush=True)
        for sid in sample_ids:
            if max_episodes is not None and total_samples >= max_episodes:
                return total_tasks, total_samples
            staged_sid = f"{task_name}__{sid}"
            if staged_sid in seen_staged_ids:
                raise RuntimeError(f"Duplicate staged sample id: {staged_sid}")
            seen_staged_ids.add(staged_sid)

            _link_file(
                task_dir / "actions" / f"{sid}.pt",
                actions_dir / f"{staged_sid}.pt",
                link_mode=link_mode,
            )
            _link_file(
                task_dir / "qpos" / f"{sid}.pt",
                qpos_dir / f"{staged_sid}.pt",
                link_mode=link_mode,
            )
            _link_file(
                task_dir / "videos" / f"{sid}.mp4",
                videos_dir / f"{staged_sid}.mp4",
                link_mode=link_mode,
            )
            _link_file(
                task_dir / "metas" / f"{sid}.txt",
                metas_dir / f"{staged_sid}.txt",
                link_mode=link_mode,
            )
            total_samples += 1
            if stage_progress_every > 0 and (total_samples % stage_progress_every == 0):
                print(
                    f"[INFO] Staged {total_samples} samples so far...",
                    flush=True,
                )

    if total_samples == 0:
        raise RuntimeError("No samples were staged from any task directory.")
    return total_tasks, total_samples


def _build_forward_cmd(
    converter_path: Path,
    staged_task_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(converter_path),
        "--input-task-dir",
        str(staged_task_dir),
        "--output-dir",
        str(output_dir),
        "--resize-width",
        str(args.resize_width),
        "--resize-height",
        str(args.resize_height),
        "--video-codec",
        str(args.video_codec),
        "--video-pix-fmt",
        str(args.video_pix_fmt),
        "--gop",
        str(args.gop),
        "--chunk-size-episodes",
        str(args.chunk_size_episodes),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.max_episodes is not None:
        cmd += ["--max-episodes", str(args.max_episodes)]
    cmd.append("--strict-validate" if args.strict_validate else "--no-strict-validate")
    return cmd


def main() -> None:
    args = _parse_args()

    if args.resize_width <= 0 or args.resize_height <= 0:
        raise ValueError("resize width/height must be positive.")
    if args.gop <= 0:
        raise ValueError("--gop must be > 0.")
    if args.chunk_size_episodes <= 0:
        raise ValueError("--chunk-size-episodes must be > 0.")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be > 0.")
    if args.stage_progress_every == 0:
        # Allow 0 as alias of disable while keeping behavior explicit.
        args.stage_progress_every = -1

    converter_path = Path(__file__).resolve().parent / "convert_egodex_task_to_lerobot_v3.py"
    if not converter_path.exists():
        raise FileNotFoundError(f"Cannot find converter script: {converter_path}")

    input_root_dir: Path = args.input_root_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    work_dir: Path = (
        args.work_dir.resolve()
        if args.work_dir is not None
        else output_dir.parent.resolve()
    )
    stage_name = args.stage_name if args.stage_name else f".tmp_egodex_stage_{os.getpid()}"
    stage_root: Path = work_dir / stage_name
    staged_task_dir: Path = stage_root / "combined_task"

    if stage_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Stage dir already exists: {stage_root}. Use --overwrite.")
        _safe_rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input root:  {input_root_dir}", flush=True)
    print(f"[INFO] Output dir:  {output_dir}", flush=True)
    print(f"[INFO] Stage dir:   {stage_root}", flush=True)
    print(f"[INFO] Link mode:   {args.link_mode}", flush=True)

    try:
        total_tasks, total_samples = _stage_combined_task(
            input_root_dir=input_root_dir,
            stage_combined_task_dir=staged_task_dir,
            link_mode=args.link_mode,
            max_episodes=args.max_episodes,
            stage_progress_every=args.stage_progress_every,
        )
        print(f"[INFO] Staged tasks:   {total_tasks}", flush=True)
        print(f"[INFO] Staged samples: {total_samples}", flush=True)

        cmd = _build_forward_cmd(
            converter_path=converter_path,
            staged_task_dir=staged_task_dir,
            output_dir=output_dir,
            args=args,
        )
        print("[INFO] Running converter:", flush=True)
        print("       " + " ".join(cmd), flush=True)
        child_env = dict(os.environ)
        child_env["PYTHONUNBUFFERED"] = "1"
        subprocess.run(cmd, check=True, env=child_env)
        print("[INFO] Batch conversion finished.", flush=True)
    finally:
        if args.keep_stage:
            print(f"[INFO] Keeping stage dir for debug: {stage_root}", flush=True)
        else:
            _safe_rmtree(stage_root)


if __name__ == "__main__":
    main()
