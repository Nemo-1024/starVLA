#!/usr/bin/env python3
"""
Build a LeRobot v3.0-style dataset from raw human egocentric videos.

The generated dataset uses a single state/action key:
  - state.xyz_rotation_6d_gripper (10-dim)
  - action.xyz_rotation_6d_gripper (10-dim)

State/action values are filled with zeros by default.
"""

from __future__ import annotations

import argparse
import fractions
import json
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


STATE_KEY = "state.xyz_rotation_6d_gripper"
ACTION_KEY = "action.xyz_rotation_6d_gripper"
VIDEO_KEY = "video.primary_view"
DEFAULT_VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".flv",
    ".ts",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw mp4 videos into a human_ego_fpv LeRobot dataset."
    )
    parser.add_argument(
        "--raw_video_dir",
        type=Path,
        required=True,
        help="Root directory (recursively scanned) that contains source videos.",
    )
    parser.add_argument(
        "--output_dataset_dir",
        type=Path,
        required=True,
        help="Output dataset directory, e.g. /path/to/datasets/human_ego_fpv_dataset",
    )
    parser.add_argument(
        "--task_text",
        type=str,
        default="Human egocentric video understanding",
        help="Task description stored in tasks.parquet.",
    )
    parser.add_argument(
        "--state_modality_name",
        type=str,
        default="xyz_rotation_6d_gripper",
        help="State subkey name in modality.json, e.g. dummy_state.",
    )
    parser.add_argument(
        "--action_modality_name",
        type=str,
        default="xyz_rotation_6d_gripper",
        help="Action subkey name in modality.json, e.g. dummy_action.",
    )
    parser.add_argument(
        "--copy_mode",
        type=str,
        default="copy",
        choices=["copy", "symlink", "hardlink"],
        help="How to place source videos into dataset video directory.",
    )
    parser.add_argument(
        "--default_fps",
        type=float,
        default=30.0,
        help="Fallback fps when source metadata is unavailable.",
    )
    parser.add_argument(
        "--resize_hw",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Optional output resolution. Example: --resize_hw 540 960",
    )
    parser.add_argument(
        "--target_fps",
        type=float,
        default=10.0,
        help="Target output FPS for frame skipping while preserving physical time. Set <=0 to disable fps resampling.",
    )
    parser.add_argument(
        "--encode_h264",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Encode output videos as h264 (default: true). Use --no-encode_h264 to keep source codec when not resizing.",
    )
    parser.add_argument(
        "--h264_crf",
        type=int,
        default=23,
        help="CRF for libx264 encoding (lower is higher quality, default: 23).",
    )
    parser.add_argument(
        "--h264_preset",
        type=str,
        default="medium",
        help="libx264 preset, e.g. ultrafast/fast/medium/slow (default: medium).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output directory if it exists.",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Optional cap on number of videos after sorting.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=100,
        help="Print one progress line every N completed videos. Set <=0 to disable.",
    )
    parser.add_argument(
        "--write_stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write meta/stats_gr00t.json after conversion.",
    )
    parser.add_argument(
        "--video_extensions",
        type=str,
        default=",".join(ext.lstrip(".") for ext in DEFAULT_VIDEO_EXTENSIONS),
        help=(
            "Comma-separated video extensions to include during recursive scan. "
            f"Default: {','.join(ext.lstrip('.') for ext in DEFAULT_VIDEO_EXTENSIONS)}"
        ),
    )
    return parser.parse_args()


def _parse_video_extensions(ext_string: str) -> set[str]:
    exts: set[str] = set()
    for item in ext_string.split(","):
        cleaned = item.strip().lower()
        if not cleaned:
            continue
        if not cleaned.startswith("."):
            cleaned = f".{cleaned}"
        exts.add(cleaned)
    if not exts:
        raise ValueError("No valid video extensions parsed from --video_extensions.")
    return exts


def _collect_videos(raw_video_dir: Path, video_extensions: set[str]) -> list[Path]:
    if raw_video_dir.is_file():
        if raw_video_dir.suffix.lower() in video_extensions:
            return [raw_video_dir]
        raise ValueError(
            f"Input path is a file but extension {raw_video_dir.suffix} is not in --video_extensions={sorted(video_extensions)}"
        )
    if not raw_video_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {raw_video_dir}")
    if not raw_video_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory for --raw_video_dir, got: {raw_video_dir}")

    videos = sorted(
        (p for p in raw_video_dir.rglob("*") if p.is_file() and p.suffix.lower() in video_extensions),
        key=lambda p: str(p.relative_to(raw_video_dir)),
    )
    if not videos:
        raise FileNotFoundError(
            f"No videos found under: {raw_video_dir} with extensions {sorted(video_extensions)}"
        )
    return videos


def _ensure_clean_output(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {out_dir}. Use --overwrite to replace."
            )
        shutil.rmtree(out_dir)
    (out_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out_dir / "videos" / "chunk-000" / VIDEO_KEY).mkdir(parents=True, exist_ok=True)


def _safe_parse_fps(rate: str, default_fps: float) -> float:
    try:
        frac = fractions.Fraction(rate)
        fps = float(frac)
    except Exception:
        fps = 0.0
    if fps <= 0:
        fps = float(default_fps)
    return fps


def _probe_video_ffprobe(video_path: Path, default_fps: float) -> tuple[int, int, int, float, str]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,codec_name,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream in {video_path}")
    stream = streams[0]

    width = int(stream.get("width", 0) or 0)
    height = int(stream.get("height", 0) or 0)
    fps = _safe_parse_fps(str(stream.get("avg_frame_rate", "0/1")), default_fps)
    codec = str(stream.get("codec_name", "") or "").lower()

    frame_count = 0
    nb_frames = stream.get("nb_frames", None)
    if nb_frames is not None:
        try:
            frame_count = int(nb_frames)
        except Exception:
            frame_count = 0
    if frame_count <= 0:
        try:
            duration = float(stream.get("duration", 0.0) or 0.0)
        except Exception:
            duration = 0.0
        if duration > 0 and fps > 0:
            frame_count = int(round(duration * fps))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video resolution for {video_path}: ({width}, {height})")
    if frame_count <= 0:
        frame_count, _, _, _ = _probe_video_cv2(video_path, default_fps)
    return frame_count, width, height, fps, codec


def _probe_video_cv2(video_path: Path, default_fps: float) -> tuple[int, int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if frame_count <= 0:
        raise ValueError(f"Video has zero frames: {video_path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video resolution for {video_path}: ({width}, {height})")
    if fps <= 0:
        fps = float(default_fps)
    return frame_count, width, height, fps


def _transcode_video_ffmpeg(
    src: Path,
    dst: Path,
    resize_hw: Optional[tuple[int, int]],
    target_fps: Optional[float],
    crf: int,
    preset: str,
) -> int:
    if dst.exists():
        dst.unlink()

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-an",
    ]
    filters: list[str] = []
    if target_fps is not None and target_fps > 0:
        filters.append(f"fps={target_fps}")
    if resize_hw is not None:
        out_h, out_w = resize_hw
        filters.append(f"scale={out_w}:{out_h}:flags=lanczos")
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg transcode failed for {src}: {(exc.stderr or exc.stdout or '').strip()}"
        ) from exc

    frame_count, _, _, _ = _probe_video_cv2(dst, default_fps=30.0)
    if frame_count <= 0:
        raise RuntimeError(f"ffmpeg wrote zero-frame file: {dst}")
    return frame_count


def _place_video(
    src: Path,
    dst: Path,
    mode: str,
    resize_hw: Optional[tuple[int, int]] = None,
    target_fps: Optional[float] = 10.0,
    encode_h264: bool = True,
    h264_crf: int = 23,
    h264_preset: str = "medium",
) -> int:
    if dst.exists():
        dst.unlink()

    if encode_h264 or resize_hw is not None or (target_fps is not None and target_fps > 0):
        return _transcode_video_ffmpeg(
            src=src,
            dst=dst,
            resize_hw=resize_hw,
            target_fps=target_fps,
            crf=h264_crf,
            preset=h264_preset,
        )

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        dst.hardlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")
    return 0


def _build_modality_json(state_modality_name: str, action_modality_name: str) -> dict:
    return {
        "state": {
            state_modality_name: {
                "start": 0,
                "end": 10,
                "absolute": True,
                "dtype": "float32",
                "original_key": STATE_KEY,
            }
        },
        "action": {
            action_modality_name: {
                "start": 0,
                "end": 10,
                "absolute": True,
                "dtype": "float32",
                "original_key": ACTION_KEY,
            }
        },
        "video": {
            "primary_view": {
                "original_key": VIDEO_KEY,
            }
        },
        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index",
            }
        },
    }


def _scalar_stats(values: np.ndarray) -> dict:
    values = values.astype(np.float64, copy=False)
    if values.size == 0:
        return {
            "mean": [0.0],
            "std": [0.0],
            "min": [0.0],
            "max": [0.0],
            "q01": [0.0],
            "q99": [0.0],
        }
    return {
        "mean": [float(np.mean(values))],
        "std": [float(np.std(values))],
        "min": [float(np.min(values))],
        "max": [float(np.max(values))],
        "q01": [float(np.quantile(values, 0.01))],
        "q99": [float(np.quantile(values, 0.99))],
    }


def _weighted_index_quantile(lengths: list[int], q: float) -> int:
    total = int(sum(lengths))
    if total <= 0:
        return 0
    rank = int(np.floor((total - 1) * q))
    cumsum = np.cumsum(np.asarray(lengths, dtype=np.int64))
    idx = int(np.searchsorted(cumsum, rank, side="right"))
    if idx < 0:
        return 0
    if idx >= len(lengths):
        return len(lengths) - 1
    return idx


def _build_stats_json(
    episode_lengths: list[int],
    episode_fps: list[float],
) -> dict:
    total_episodes = len(episode_lengths)
    total_frames = int(sum(episode_lengths))

    if total_frames > 0 and total_episodes > 0:
        sum_i = 0.0
        sum_i2 = 0.0
        for idx, length in enumerate(episode_lengths):
            l = float(max(length, 0))
            sum_i += float(idx) * l
            sum_i2 += float(idx * idx) * l
        mean_i = sum_i / float(total_frames)
        var_i = max(0.0, (sum_i2 / float(total_frames)) - mean_i * mean_i)
        episode_index_stats = {
            "mean": [float(mean_i)],
            "std": [float(np.sqrt(var_i))],
            "min": [0.0],
            "max": [float(total_episodes - 1)],
            "q01": [float(_weighted_index_quantile(episode_lengths, 0.01))],
            "q99": [float(_weighted_index_quantile(episode_lengths, 0.99))],
        }
    else:
        episode_index_stats = _scalar_stats(np.asarray([], dtype=np.float64))

    if total_frames > 0:
        sum_t = 0.0
        sum_t2 = 0.0
        max_t = 0.0
        for length, fps in zip(episode_lengths, episode_fps, strict=True):
            if length <= 0 or fps <= 0:
                continue
            l = float(length)
            ff = float(fps)
            sum_t += ((l - 1.0) * l) / (2.0 * ff)
            sum_t2 += ((l - 1.0) * l * (2.0 * l - 1.0)) / (6.0 * ff * ff)
            max_t = max(max_t, (l - 1.0) / ff)
        mean_t = sum_t / float(total_frames)
        var_t = max(0.0, (sum_t2 / float(total_frames)) - mean_t * mean_t)
        timestamp_stats = {
            "mean": [float(mean_t)],
            "std": [float(np.sqrt(var_t))],
            "min": [0.0],
            "max": [float(max_t)],
            "q01": [0.0],
            "q99": [float(max_t)],
        }
    else:
        timestamp_stats = _scalar_stats(np.asarray([], dtype=np.float64))

    task_index_stats = {
        "mean": [0.0],
        "std": [0.0],
        "min": [0.0],
        "max": [0.0],
        "q01": [0.0],
        "q99": [0.0],
    }
    zero_vec = [0.0] * 10
    zero_vec_stats = {
        "mean": zero_vec,
        "std": zero_vec,
        "min": zero_vec,
        "max": zero_vec,
        "q01": zero_vec,
        "q99": zero_vec,
    }

    return {
        "episode_index": episode_index_stats,
        "timestamp": timestamp_stats,
        "task_index": task_index_stats,
        STATE_KEY: zero_vec_stats,
        ACTION_KEY: zero_vec_stats,
    }


def _build_info_json(
    total_episodes: int,
    total_frames: int,
    width: int,
    height: int,
    fps: float,
    video_codec: str,
) -> dict:
    return {
        "codebase_version": "v3.0",
        "robot_type": "human_ego_fpv",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {
            "train": f"0:{total_episodes}",
        },
        "data_path": "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet",
        "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4",
        "features": {
            STATE_KEY: {
                "dtype": "float32",
                "shape": [10],
            },
            ACTION_KEY: {
                "dtype": "float32",
                "shape": [10],
            },
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": height,
                    "video.width": width,
                    "video.codec": video_codec,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
        },
    }


def main() -> None:
    args = _parse_args()
    video_extensions = _parse_video_extensions(args.video_extensions)
    videos = _collect_videos(args.raw_video_dir, video_extensions)
    if args.max_videos is not None:
        if args.max_videos <= 0:
            raise ValueError(f"--max_videos must be > 0, got {args.max_videos}")
        videos = videos[: args.max_videos]
    _ensure_clean_output(args.output_dataset_dir, args.overwrite)
    print(
        f"Found {len(videos)} video files under {args.raw_video_dir} "
        f"(extensions={sorted(video_extensions)})"
    )

    resize_hw: Optional[tuple[int, int]] = None
    if args.resize_hw is not None:
        out_h, out_w = int(args.resize_hw[0]), int(args.resize_hw[1])
        if out_h <= 0 or out_w <= 0:
            raise ValueError(f"resize_hw must be positive, got {(out_h, out_w)}")
        resize_hw = (out_h, out_w)
    target_fps: Optional[float] = float(args.target_fps)
    if target_fps <= 0:
        target_fps = None

    tasks_df = pd.DataFrame([{"task_index": 0, "task": args.task_text}])
    tasks_df.to_parquet(args.output_dataset_dir / "meta" / "tasks.parquet", index=False)

    episodes_rows: list[dict] = []
    total_frames = 0
    first_width = 0
    first_height = 0
    first_fps = float(args.default_fps)
    first_codec = "h264"
    from_index = 0
    episode_lengths: list[int] = []
    episode_fps: list[float] = []

    video_iter = tqdm(
        enumerate(videos),
        total=len(videos),
        desc="Processing videos",
        unit="video",
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=0.5,
        leave=True,
    )
    for episode_index, src_video in video_iter:
        try:
            display_name = str(src_video.relative_to(args.raw_video_dir))
        except Exception:
            display_name = src_video.name
        video_iter.set_postfix_str(display_name, refresh=False)

        frame_count, width, height, fps, src_codec = _probe_video_ffprobe(src_video, args.default_fps)
        if resize_hw is not None:
            height, width = resize_hw

        dst_video = (
            args.output_dataset_dir
            / "videos"
            / "chunk-000"
            / VIDEO_KEY
            / f"episode_{episode_index:06d}.mp4"
        )
        _place_video(
            src=src_video,
            dst=dst_video,
            mode=args.copy_mode,
            resize_hw=resize_hw,
            target_fps=target_fps,
            encode_h264=args.encode_h264,
            h264_crf=args.h264_crf,
            h264_preset=args.h264_preset,
        )
        if args.encode_h264 or resize_hw is not None or target_fps is not None:
            frame_count, width, height, fps, codec = _probe_video_ffprobe(
                dst_video, args.default_fps
            )
            if codec != "h264":
                raise RuntimeError(f"Expected h264 output but got codec={codec} for {dst_video}")
        else:
            codec = src_codec

        if episode_index == 0:
            first_width, first_height, first_fps, first_codec = width, height, fps, codec
        if frame_count <= 0:
            raise RuntimeError(f"No frames were written for {src_video}")

        timestamps = (np.arange(frame_count, dtype=np.float64) / fps).tolist()
        zeros = [np.zeros(10, dtype=np.float32) for _ in range(frame_count)]
        frame_df = pd.DataFrame(
            {
                "episode_index": np.full(frame_count, episode_index, dtype=np.int64),
                "timestamp": timestamps,
                "task_index": np.zeros(frame_count, dtype=np.int64),
                STATE_KEY: zeros,
                ACTION_KEY: zeros,
            }
        )
        frame_df.to_parquet(
            args.output_dataset_dir / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet",
            index=False,
        )

        to_index = from_index + frame_count
        episodes_rows.append(
            {
                "episode_index": episode_index,
                "length": frame_count,
                "dataset_from_index": from_index,
                "dataset_to_index": to_index,
                "data/chunk_index": 0,
                f"videos/{VIDEO_KEY}/chunk_index": 0,
                f"videos/{VIDEO_KEY}/file_index": episode_index,
                f"videos/{VIDEO_KEY}/from_timestamp": 0.0,
            }
        )
        from_index = to_index
        total_frames += frame_count
        episode_lengths.append(frame_count)
        episode_fps.append(float(fps))
        if args.progress_every > 0 and (
            (episode_index + 1) % args.progress_every == 0 or (episode_index + 1) == len(videos)
        ):
            video_iter.write(
                f"[{episode_index + 1}/{len(videos)}] completed: {display_name} "
                f"(frames={frame_count}, fps={fps:.3f})"
            )

    episodes_df = pd.DataFrame(episodes_rows)
    episodes_df.to_parquet(
        args.output_dataset_dir / "meta" / "episodes" / "chunk-000" / "episodes_000.parquet",
        index=False,
    )

    modality_json = _build_modality_json(
        state_modality_name=args.state_modality_name,
        action_modality_name=args.action_modality_name,
    )
    with open(args.output_dataset_dir / "meta" / "modality.json", "w", encoding="utf-8") as f:
        json.dump(modality_json, f, indent=2)

    info_json = _build_info_json(
        total_episodes=len(videos),
        total_frames=total_frames,
        width=first_width,
        height=first_height,
        fps=first_fps,
        video_codec=first_codec,
    )
    with open(args.output_dataset_dir / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(info_json, f, indent=2)
    if args.write_stats:
        stats_json = _build_stats_json(
            episode_lengths=episode_lengths,
            episode_fps=episode_fps,
        )
        with open(args.output_dataset_dir / "meta" / "stats_gr00t.json", "w", encoding="utf-8") as f:
            json.dump(stats_json, f, indent=4)

    print(f"Built dataset at: {args.output_dataset_dir}")
    print(f"Episodes: {len(videos)}")
    print(f"Frames: {total_frames}")


if __name__ == "__main__":
    main()
