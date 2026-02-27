#!/usr/bin/env python3
"""
Convert one EgoDex task directory to a LeRobot v3.0-style dataset (standalone writer).

Expected input layout:
  <task_dir>/
    actions/*.pt
    qpos/*.pt
    videos/*.mp4
    metas/*.txt

Output layout:
  <output_dir>/
    meta/info.json
    meta/modality.json
    meta/stats_gr00t.json
    meta/tasks.parquet
    meta/episodes/chunk-XXX/file-000.parquet
    data/chunk-XXX/file-000.parquet
    videos/observation.images.ego_view/chunk-XXX/file-000.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


SOURCE_FPS = 5.0
VIDEO_KEY = "observation.images.ego_view"
STATE_KEY = "observation.state"
ACTION_KEY = "action"


@dataclass
class EpisodeSpec:
    episode_index: int
    sample_id: str
    task_text: str
    task_index: int
    qpos_path: Path
    action_path: Path
    video_path: Path
    meta_path: Path
    length: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one EgoDex task_dir into LeRobot v3.0 format."
    )
    parser.add_argument("--input-task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
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
    return parser.parse_args()


def _require_tools() -> None:
    for tool in ["ffmpeg", "ffprobe"]:
        if shutil.which(tool) is None:
            raise FileNotFoundError(f"Required tool `{tool}` not found in PATH.")


def _ensure_clean_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _sample_sort_key(sample_id: str) -> tuple[int, ...]:
    # E.g. "1_1909" -> (1, 1909), fallback to tokenized numeric ordering.
    tokens = re.split(r"(\d+)", sample_id)
    key = []
    for tok in tokens:
        if not tok:
            continue
        if tok.isdigit():
            key.append(int(tok))
        else:
            key.extend(ord(c) for c in tok)
    return tuple(key)


def _stems_for_ext(directory: Path, ext: str) -> set[str]:
    return {p.stem for p in directory.glob(f"*{ext}") if p.is_file()}


def _discover_sample_ids(task_dir: Path) -> list[str]:
    actions_dir = task_dir / "actions"
    qpos_dir = task_dir / "qpos"
    videos_dir = task_dir / "videos"
    metas_dir = task_dir / "metas"
    for d in [actions_dir, qpos_dir, videos_dir, metas_dir]:
        if not d.exists():
            raise FileNotFoundError(f"Missing required subdir: {d}")

    stems_actions = _stems_for_ext(actions_dir, ".pt")
    stems_qpos = _stems_for_ext(qpos_dir, ".pt")
    stems_videos = _stems_for_ext(videos_dir, ".mp4")
    stems_metas = _stems_for_ext(metas_dir, ".txt")
    sample_ids = stems_actions & stems_qpos & stems_videos & stems_metas
    if not sample_ids:
        raise RuntimeError(
            "No intersecting sample ids among actions/qpos/videos/metas."
        )
    return sorted(sample_ids, key=_sample_sort_key)


def _run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _probe_video(video_path: Path) -> tuple[int, float, int, int, str]:
    """
    Return (frame_count, fps, width, height, codec).
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,avg_frame_rate,width,height,codec_name",
        "-of",
        "default=noprint_wrappers=1",
        str(video_path),
    ]
    out = _run_cmd(cmd).stdout.strip().splitlines()
    kv = {}
    for line in out:
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()

    width = int(kv.get("width", "0") or 0)
    height = int(kv.get("height", "0") or 0)
    codec = kv.get("codec_name", "")
    fps_str = kv.get("avg_frame_rate", "0/1")
    if "/" in fps_str:
        a, b = fps_str.split("/", 1)
        fps = float(a) / float(b) if float(b) != 0 else 0.0
    else:
        fps = float(fps_str)

    n_str = kv.get("nb_frames", "")
    frame_count = int(n_str) if n_str and n_str != "N/A" else 0
    if frame_count <= 0:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for frame count: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
        if width <= 0:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if height <= 0:
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    if frame_count <= 0:
        raise RuntimeError(f"Invalid frame count for {video_path}")
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid resolution for {video_path}")
    if fps <= 0:
        raise RuntimeError(f"Invalid fps for {video_path}")
    return frame_count, fps, width, height, codec


def _load_state_action(qpos_path: Path, action_path: Path) -> tuple[np.ndarray, np.ndarray]:
    qpos = torch.load(qpos_path, map_location="cpu")
    action = torch.load(action_path, map_location="cpu")
    if not isinstance(qpos, torch.Tensor) or not isinstance(action, torch.Tensor):
        raise TypeError(f"Expected tensors for qpos/action, got {type(qpos)} and {type(action)}")
    q = qpos.detach().cpu().numpy()
    a = action.detach().cpu().numpy()
    if q.ndim != 2 or a.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got {q.shape} and {a.shape}")
    return q, a


def _compute_stats(values: np.ndarray) -> dict[str, list[float]]:
    if values.size == 0:
        zeros = [0.0] * 14
        return {
            "mean": zeros,
            "std": zeros,
            "min": zeros,
            "max": zeros,
            "q01": zeros,
            "q99": zeros,
        }
    return {
        "mean": np.mean(values, axis=0).astype(np.float64).tolist(),
        "std": np.std(values, axis=0).astype(np.float64).tolist(),
        "min": np.min(values, axis=0).astype(np.float64).tolist(),
        "max": np.max(values, axis=0).astype(np.float64).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).astype(np.float64).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(np.float64).tolist(),
    }


def _reencode_video(
    src: Path,
    dst: Path,
    width: int,
    height: int,
    codec: str,
    pix_fmt: str,
    gop: int,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = f"fps={SOURCE_FPS},scale={width}:{height}:flags=lanczos"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-an",
        "-vf",
        vf,
        "-c:v",
        codec,
        "-pix_fmt",
        pix_fmt,
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-sc_threshold",
        "0",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    try:
        _run_cmd(cmd)
    except subprocess.CalledProcessError as exc:
        msg = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg re-encode failed for {src}: {msg}") from exc


def _concat_videos(clip_paths: list[Path], dst_video: Path) -> None:
    if not clip_paths:
        raise ValueError("No clip paths to concatenate.")
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    list_file = dst_video.parent / f".concat_{dst_video.stem}.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in clip_paths:
            escaped = str(p.resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(dst_video),
    ]
    try:
        _run_cmd(cmd)
    except subprocess.CalledProcessError as exc:
        msg = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg concat failed for {dst_video}: {msg}") from exc
    finally:
        if list_file.exists():
            list_file.unlink()


def _build_modality_json() -> dict:
    return {
        "video": {
            "primary_view": {
                "original_key": VIDEO_KEY,
            }
        },
        "state": {
            "left_joints": {
                "start": 0,
                "end": 6,
                "absolute": True,
                "dtype": "float32",
                "original_key": STATE_KEY,
            },
            "left_gripper": {
                "start": 6,
                "end": 7,
                "absolute": True,
                "dtype": "float32",
                "original_key": STATE_KEY,
            },
            "right_joints": {
                "start": 7,
                "end": 13,
                "absolute": True,
                "dtype": "float32",
                "original_key": STATE_KEY,
            },
            "right_gripper": {
                "start": 13,
                "end": 14,
                "absolute": True,
                "dtype": "float32",
                "original_key": STATE_KEY,
            },
        },
        "action": {
            "left_joints": {
                "start": 0,
                "end": 6,
                "absolute": True,
                "dtype": "float32",
                "original_key": ACTION_KEY,
            },
            "left_gripper": {
                "start": 6,
                "end": 7,
                "absolute": True,
                "dtype": "float32",
                "original_key": ACTION_KEY,
            },
            "right_joints": {
                "start": 7,
                "end": 13,
                "absolute": True,
                "dtype": "float32",
                "original_key": ACTION_KEY,
            },
            "right_gripper": {
                "start": 13,
                "end": 14,
                "absolute": True,
                "dtype": "float32",
                "original_key": ACTION_KEY,
            },
        },
        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index",
            }
        },
    }


def _build_info_json(
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_chunks: int,
    chunk_size: int,
    width: int,
    height: int,
    video_codec: str,
    video_pix_fmt: str,
) -> dict:
    return {
        "codebase_version": "v3.0",
        "robot_type": "egodex_human_ego",
        "total_episodes": int(total_episodes),
        "total_frames": int(total_frames),
        "total_tasks": int(total_tasks),
        "total_videos": int(total_chunks),
        "total_chunks": int(total_chunks),
        "chunks_size": int(chunk_size),
        "fps": float(SOURCE_FPS),
        "splits": {
            "train": f"0:{total_episodes}",
        },
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": height,
                    "video.width": width,
                    "video.codec": video_codec,
                    "video.pix_fmt": video_pix_fmt,
                    "video.is_depth_map": False,
                    "video.fps": float(SOURCE_FPS),
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            STATE_KEY: {
                "dtype": "float32",
                "shape": [14],
                "names": {
                    "motors": [
                        "left_joint_0",
                        "left_joint_1",
                        "left_joint_2",
                        "left_joint_3",
                        "left_joint_4",
                        "left_joint_5",
                        "left_gripper",
                        "right_joint_0",
                        "right_joint_1",
                        "right_joint_2",
                        "right_joint_3",
                        "right_joint_4",
                        "right_joint_5",
                        "right_gripper",
                    ]
                },
            },
            ACTION_KEY: {
                "dtype": "float32",
                "shape": [14],
                "names": {
                    "motors": [
                        "left_joint_0",
                        "left_joint_1",
                        "left_joint_2",
                        "left_joint_3",
                        "left_joint_4",
                        "left_joint_5",
                        "left_gripper",
                        "right_joint_0",
                        "right_joint_1",
                        "right_joint_2",
                        "right_joint_3",
                        "right_joint_4",
                        "right_joint_5",
                        "right_gripper",
                    ]
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


def _chunks(items: list[EpisodeSpec], n: int) -> Iterable[list[EpisodeSpec]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _validate_episode_spec(
    task_dir: Path,
    sample_id: str,
    strict_validate: bool,
) -> tuple[bool, str, int]:
    qpos_path = task_dir / "qpos" / f"{sample_id}.pt"
    action_path = task_dir / "actions" / f"{sample_id}.pt"
    video_path = task_dir / "videos" / f"{sample_id}.mp4"
    meta_path = task_dir / "metas" / f"{sample_id}.txt"
    q, a = _load_state_action(qpos_path, action_path)
    n_frames, fps, _, _, _ = _probe_video(video_path)
    msg = []

    ok = True
    if q.shape[1] != 14 or a.shape[1] != 14:
        ok = False
        msg.append(f"expected 14 dims, got qpos={q.shape}, action={a.shape}")
    if q.shape[0] != a.shape[0] + 1:
        ok = False
        msg.append(f"len mismatch qpos={q.shape[0]} vs action+1={a.shape[0] + 1}")
    if n_frames != q.shape[0]:
        ok = False
        msg.append(f"video frames={n_frames} vs qpos len={q.shape[0]}")
    if not math.isclose(fps, SOURCE_FPS, rel_tol=1e-5, abs_tol=1e-5):
        ok = False
        msg.append(f"fps={fps} expected={SOURCE_FPS}")

    if not ok and strict_validate:
        raise ValueError(f"Invalid sample `{sample_id}`: " + "; ".join(msg))
    if not ok:
        print(f"[WARN] Skip invalid sample `{sample_id}`: " + "; ".join(msg))
        return False, "", 0

    task_text = meta_path.read_text(encoding="utf-8").strip()
    return True, task_text, int(q.shape[0])


def _post_validate_outputs(
    output_dir: Path,
    total_episodes: int,
    total_frames: int,
) -> None:
    required = [
        output_dir / "meta" / "info.json",
        output_dir / "meta" / "modality.json",
        output_dir / "meta" / "stats_gr00t.json",
        output_dir / "meta" / "tasks.parquet",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing output file: {p}")

    info = json.loads((output_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    if int(info.get("total_episodes", -1)) != total_episodes:
        raise ValueError("info.json total_episodes mismatch")
    if int(info.get("total_frames", -1)) != total_frames:
        raise ValueError("info.json total_frames mismatch")

    tasks_df = pd.read_parquet(output_dir / "meta" / "tasks.parquet")
    if "task_index" not in tasks_df.columns or "task" not in tasks_df.columns:
        raise ValueError("tasks.parquet missing required columns")

    ep_files = sorted((output_dir / "meta" / "episodes").glob("chunk-*/file-000.parquet"))
    data_files = sorted((output_dir / "data").glob("chunk-*/file-000.parquet"))
    if not ep_files or not data_files:
        raise FileNotFoundError("Missing chunk parquet files under data/ or meta/episodes/.")

    # Row-level and video-time checks.
    for ep_file in ep_files:
        ep_df = pd.read_parquet(ep_file)
        if not ep_df.empty:
            for _, row in ep_df.iterrows():
                if int(row["dataset_to_index"]) <= int(row["dataset_from_index"]):
                    raise ValueError(f"Invalid dataset index range in {ep_file}")
                if float(row[f"videos/{VIDEO_KEY}/to_timestamp"]) <= float(row[f"videos/{VIDEO_KEY}/from_timestamp"]):
                    raise ValueError(f"Invalid video timestamp range in {ep_file}")

            # contiguity
            rows = ep_df.sort_values("episode_index")
            prev_to = None
            for _, row in rows.iterrows():
                fr = float(row[f"videos/{VIDEO_KEY}/from_timestamp"])
                to = float(row[f"videos/{VIDEO_KEY}/to_timestamp"])
                if prev_to is not None and not math.isclose(fr, prev_to, abs_tol=1e-6):
                    raise ValueError(f"Non-contiguous from/to timestamp in {ep_file}")
                prev_to = to

    # Data checks.
    for data_file in data_files:
        df = pd.read_parquet(data_file)
        if df.empty:
            continue
        for ep_idx, grp in df.groupby("episode_index", sort=False):
            ts = grp["timestamp"].to_numpy(np.float64)
            if np.any(np.diff(ts) < -1e-8):
                raise ValueError(f"timestamp is not monotonic for episode_index={ep_idx}")
            if len(grp) >= 2:
                last_action = np.asarray(grp.iloc[-1][ACTION_KEY], dtype=np.float32)
                prev_action = np.asarray(grp.iloc[-2][ACTION_KEY], dtype=np.float32)
                if not np.allclose(last_action, prev_action):
                    raise ValueError(
                        f"last frame action is not repeated for episode_index={ep_idx}"
                    )


def main() -> None:
    args = _parse_args()
    _require_tools()

    if args.resize_width <= 0 or args.resize_height <= 0:
        raise ValueError("resize width/height must be positive.")
    if args.gop <= 0:
        raise ValueError("--gop must be > 0.")
    if args.chunk_size_episodes <= 0:
        raise ValueError("--chunk-size-episodes must be > 0.")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be > 0.")

    input_task_dir: Path = args.input_task_dir
    output_dir: Path = args.output_dir

    if not input_task_dir.exists():
        raise FileNotFoundError(f"Input task dir does not exist: {input_task_dir}")
    _ensure_clean_output(output_dir, args.overwrite)

    sample_ids = _discover_sample_ids(input_task_dir)
    if args.max_episodes is not None:
        sample_ids = sample_ids[: args.max_episodes]
    if not sample_ids:
        raise RuntimeError("No samples selected for conversion.")

    print(f"Discovered {len(sample_ids)} samples to convert.")

    # First pass: validate and gather task texts.
    valid_samples: list[tuple[str, str, int]] = []
    for sid in tqdm(sample_ids, desc="Validating samples", unit="sample"):
        ok, task_text, length = _validate_episode_spec(
            input_task_dir, sid, strict_validate=args.strict_validate
        )
        if ok:
            valid_samples.append((sid, task_text, length))

    if not valid_samples:
        raise RuntimeError("No valid samples after validation.")

    # Deterministic dedup based on first occurrence in sample-id order.
    task_to_index: dict[str, int] = {}
    tasks_rows = []
    for _, task_text, _ in valid_samples:
        if task_text not in task_to_index:
            t_idx = len(task_to_index)
            task_to_index[task_text] = t_idx
            tasks_rows.append({"task_index": t_idx, "task": task_text})

    episodes: list[EpisodeSpec] = []
    for ep_idx, (sid, task_text, length) in enumerate(valid_samples):
        episodes.append(
            EpisodeSpec(
                episode_index=ep_idx,
                sample_id=sid,
                task_text=task_text,
                task_index=task_to_index[task_text],
                qpos_path=input_task_dir / "qpos" / f"{sid}.pt",
                action_path=input_task_dir / "actions" / f"{sid}.pt",
                video_path=input_task_dir / "videos" / f"{sid}.mp4",
                meta_path=input_task_dir / "metas" / f"{sid}.txt",
                length=length,
            )
        )

    # Prepare directories.
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / VIDEO_KEY).mkdir(parents=True, exist_ok=True)
    tmp_root = output_dir / ".tmp_convert"
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Write tasks parquet now.
    tasks_df = pd.DataFrame(tasks_rows)
    tasks_df.to_parquet(output_dir / "meta" / "tasks.parquet", index=False)

    state_collect = []
    action_collect = []

    global_row_index = 0
    total_frames = 0
    total_chunks = int(math.ceil(len(episodes) / args.chunk_size_episodes))

    chunk_iter = list(_chunks(episodes, args.chunk_size_episodes))
    for chunk_idx, chunk_eps in enumerate(tqdm(chunk_iter, desc="Converting chunks", unit="chunk")):
        data_chunk_dir = output_dir / "data" / f"chunk-{chunk_idx:03d}"
        ep_chunk_dir = output_dir / "meta" / "episodes" / f"chunk-{chunk_idx:03d}"
        vid_chunk_dir = output_dir / "videos" / VIDEO_KEY / f"chunk-{chunk_idx:03d}"
        data_chunk_dir.mkdir(parents=True, exist_ok=True)
        ep_chunk_dir.mkdir(parents=True, exist_ok=True)
        vid_chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_tmp_dir = tmp_root / f"chunk-{chunk_idx:03d}"
        chunk_tmp_dir.mkdir(parents=True, exist_ok=True)

        clip_paths: list[Path] = []
        episodes_rows = []
        col_state = []
        col_action = []
        col_timestamp = []
        col_frame_index = []
        col_episode_index = []
        col_index = []
        col_task_index = []

        chunk_video_offset = 0.0
        for local_i, ep in enumerate(chunk_eps):
            clip_path = chunk_tmp_dir / f"episode_{ep.episode_index:06d}.mp4"
            _reencode_video(
                ep.video_path,
                clip_path,
                width=args.resize_width,
                height=args.resize_height,
                codec=args.video_codec,
                pix_fmt=args.video_pix_fmt,
                gop=args.gop,
            )
            clip_frames, clip_fps, clip_w, clip_h, _ = _probe_video(clip_path)
            if args.strict_validate:
                if clip_frames != ep.length:
                    raise ValueError(
                        f"Re-encoded clip frame mismatch for {ep.sample_id}: "
                        f"{clip_frames} vs expected {ep.length}"
                    )
                if not math.isclose(clip_fps, SOURCE_FPS, rel_tol=1e-5, abs_tol=1e-5):
                    raise ValueError(
                        f"Re-encoded clip fps mismatch for {ep.sample_id}: {clip_fps}"
                    )
                if clip_w != args.resize_width or clip_h != args.resize_height:
                    raise ValueError(
                        f"Re-encoded clip shape mismatch for {ep.sample_id}: "
                        f"{clip_w}x{clip_h}"
                    )
            clip_paths.append(clip_path)

            qpos, action = _load_state_action(ep.qpos_path, ep.action_path)
            qpos = qpos.astype(np.float32, copy=False)
            action = action.astype(np.float32, copy=False)

            if action.shape[0] == 0:
                # Fallback path if ever encountered.
                action_pad = np.zeros_like(qpos, dtype=np.float32)
            else:
                action_pad = np.concatenate([action, action[-1:, :]], axis=0)

            if args.strict_validate:
                if action_pad.shape != qpos.shape:
                    raise ValueError(
                        f"Action pad shape mismatch for {ep.sample_id}: "
                        f"{action_pad.shape} vs {qpos.shape}"
                    )

            T = qpos.shape[0]
            col_state.extend([row for row in qpos])
            col_action.extend([row for row in action_pad])
            col_timestamp.extend((np.arange(T, dtype=np.float32) / np.float32(SOURCE_FPS)).tolist())
            col_frame_index.extend(np.arange(T, dtype=np.int64).tolist())
            col_episode_index.extend([ep.episode_index] * T)
            col_index.extend(np.arange(global_row_index, global_row_index + T, dtype=np.int64).tolist())
            col_task_index.extend([ep.task_index] * T)

            state_collect.append(qpos)
            action_collect.append(action_pad)

            dataset_from_index = global_row_index
            dataset_to_index = global_row_index + T
            from_ts = chunk_video_offset
            to_ts = chunk_video_offset + (T / SOURCE_FPS)
            episodes_rows.append(
                {
                    "episode_index": ep.episode_index,
                    "tasks": [ep.task_text],
                    "length": T,
                    "data/chunk_index": chunk_idx,
                    "data/file_index": 0,
                    "dataset_from_index": dataset_from_index,
                    "dataset_to_index": dataset_to_index,
                    f"videos/{VIDEO_KEY}/chunk_index": chunk_idx,
                    f"videos/{VIDEO_KEY}/file_index": 0,
                    f"videos/{VIDEO_KEY}/from_timestamp": float(from_ts),
                    f"videos/{VIDEO_KEY}/to_timestamp": float(to_ts),
                    "source_sample_id": ep.sample_id,
                }
            )

            global_row_index += T
            total_frames += T
            chunk_video_offset = to_ts

        # Write chunk data parquet.
        data_df = pd.DataFrame(
            {
                STATE_KEY: col_state,
                ACTION_KEY: col_action,
                "timestamp": np.asarray(col_timestamp, dtype=np.float32),
                "frame_index": np.asarray(col_frame_index, dtype=np.int64),
                "episode_index": np.asarray(col_episode_index, dtype=np.int64),
                "index": np.asarray(col_index, dtype=np.int64),
                "task_index": np.asarray(col_task_index, dtype=np.int64),
            }
        )
        data_df.to_parquet(data_chunk_dir / "file-000.parquet", index=False)

        # Write chunk episode metadata parquet.
        ep_df = pd.DataFrame(episodes_rows)
        ep_df.to_parquet(ep_chunk_dir / "file-000.parquet", index=False)

        # Concatenate chunk video.
        dst_chunk_video = vid_chunk_dir / "file-000.mp4"
        _concat_videos(clip_paths, dst_chunk_video)

        if args.strict_validate:
            merged_frames, merged_fps, merged_w, merged_h, _ = _probe_video(dst_chunk_video)
            expected_frames = int(sum(ep.length for ep in chunk_eps))
            if merged_frames != expected_frames:
                raise ValueError(
                    f"Chunk video frame count mismatch for chunk-{chunk_idx:03d}: "
                    f"{merged_frames} vs expected {expected_frames}"
                )
            if not math.isclose(merged_fps, SOURCE_FPS, rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(f"Chunk video fps mismatch: {merged_fps}")
            if merged_w != args.resize_width or merged_h != args.resize_height:
                raise ValueError(f"Chunk video resolution mismatch: {merged_w}x{merged_h}")

        shutil.rmtree(chunk_tmp_dir, ignore_errors=True)

    # Cleanup temporary root.
    shutil.rmtree(tmp_root, ignore_errors=True)

    # Write metadata files.
    modality = _build_modality_json()
    with open(output_dir / "meta" / "modality.json", "w", encoding="utf-8") as f:
        json.dump(modality, f, indent=2)

    info = _build_info_json(
        total_episodes=len(episodes),
        total_frames=total_frames,
        total_tasks=len(task_to_index),
        total_chunks=total_chunks,
        chunk_size=args.chunk_size_episodes,
        width=args.resize_width,
        height=args.resize_height,
        video_codec=args.video_codec,
        video_pix_fmt=args.video_pix_fmt,
    )
    with open(output_dir / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    state_all = np.concatenate(state_collect, axis=0) if state_collect else np.zeros((0, 14), dtype=np.float32)
    action_all = np.concatenate(action_collect, axis=0) if action_collect else np.zeros((0, 14), dtype=np.float32)
    stats = {
        STATE_KEY: _compute_stats(state_all),
        ACTION_KEY: _compute_stats(action_all),
    }
    with open(output_dir / "meta" / "stats_gr00t.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    _post_validate_outputs(
        output_dir=output_dir,
        total_episodes=len(episodes),
        total_frames=total_frames,
    )

    print("\nConversion complete.")
    print(f"Input task dir:  {input_task_dir}")
    print(f"Output dataset:  {output_dir}")
    print(f"Episodes:        {len(episodes)}")
    print(f"Frames:          {total_frames}")
    print(f"Tasks:           {len(task_to_index)}")
    print(f"Chunks:          {total_chunks}")


if __name__ == "__main__":
    main()
