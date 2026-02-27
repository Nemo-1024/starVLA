#!/usr/bin/env python3
"""Benchmark dataloader throughput for different timestamp frame-fetch strategies.

Example:
  python latent_action_model/scripts/benchmark_dataloader_video_strategies.py \
    --config latent_action_model/config/dino_base_ae.yaml \
    --data-mix droid \
    --strategy two_seek \
    --steps 80 \
    --warmup 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import av
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_data_cfg(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid yaml root type: {type(cfg)}")
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("Config must contain a dict field: data")
    return dict(data_cfg)


def _to_float64_1d(timestamps: list[float] | np.ndarray) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size == 0:
        raise ValueError("timestamps must be non-empty")
    return ts


def _two_seek_get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    video_backend: str = "pyav",
    video_backend_kwargs: dict | None = None,
) -> np.ndarray:
    del video_backend_kwargs
    if video_backend != "pyav":
        raise NotImplementedError(f"Only pyav backend is supported, got {video_backend}")
    ts = _to_float64_1d(timestamps)

    unique_ts, inverse = np.unique(ts, return_inverse=True)
    selected_unique: list[np.ndarray] = []

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        if time_base <= 0:
            raise RuntimeError(f"Invalid time_base for video {video_path}: {time_base}")

        for target_ts in unique_ts.tolist():
            seek_ts = max(float(target_ts), 0.0)
            seek_pts = int(seek_ts / time_base)
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)

            prev_frame = None
            prev_ts = None
            chosen = None

            for frame in container.decode(video=0):
                if frame.pts is None:
                    continue
                current_ts = float(frame.pts * time_base)
                frame_rgb = frame.to_ndarray(format="rgb24")

                if current_ts < target_ts:
                    prev_frame = frame_rgb
                    prev_ts = current_ts
                    continue

                if prev_frame is None:
                    chosen = frame_rgb
                else:
                    if abs(prev_ts - target_ts) <= abs(current_ts - target_ts):
                        chosen = prev_frame
                    else:
                        chosen = frame_rgb
                break

            if chosen is None:
                if prev_frame is not None:
                    chosen = prev_frame
                else:
                    raise RuntimeError(
                        f"No frames loaded from {video_path} for timestamp={target_ts}."
                    )
            selected_unique.append(chosen)

    selected = [selected_unique[int(i)] for i in inverse.tolist()]
    return np.asarray(selected)


def _single_seek_get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    video_backend: str = "pyav",
    video_backend_kwargs: dict | None = None,
) -> np.ndarray:
    del video_backend_kwargs
    if video_backend != "pyav":
        raise NotImplementedError(f"Only pyav backend is supported, got {video_backend}")
    ts = _to_float64_1d(timestamps)

    unique_ts, inverse = np.unique(ts, return_inverse=True)
    selected_unique: list[np.ndarray | None] = [None] * len(unique_ts)

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        if time_base <= 0:
            raise RuntimeError(f"Invalid time_base for video {video_path}: {time_base}")

        min_target_ts = float(max(0.0, unique_ts[0]))
        seek_pts = int(min_target_ts / time_base)
        container.seek(seek_pts, stream=stream, backward=True, any_frame=False)

        target_idx = 0
        prev_frame = None
        prev_ts = None
        last_frame = None

        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            current_ts = float(frame.pts * time_base)
            frame_rgb = frame.to_ndarray(format="rgb24")
            last_frame = frame_rgb

            while target_idx < len(unique_ts) and current_ts >= unique_ts[target_idx]:
                if prev_frame is None or prev_ts is None:
                    chosen = frame_rgb
                else:
                    if abs(prev_ts - unique_ts[target_idx]) <= abs(current_ts - unique_ts[target_idx]):
                        chosen = prev_frame
                    else:
                        chosen = frame_rgb
                selected_unique[target_idx] = chosen
                target_idx += 1

            prev_frame = frame_rgb
            prev_ts = current_ts
            if target_idx >= len(unique_ts):
                break

        if target_idx < len(unique_ts):
            if prev_frame is None:
                raise RuntimeError(f"No frames decoded from {video_path}")
            fallback = prev_frame if last_frame is None else last_frame
            for i in range(target_idx, len(unique_ts)):
                selected_unique[i] = fallback

    selected_unique_np = [f for f in selected_unique if f is not None]
    if len(selected_unique_np) != len(unique_ts):
        raise RuntimeError("Internal error: unresolved target timestamps remain")
    selected = [selected_unique_np[int(i)] for i in inverse.tolist()]
    return np.asarray(selected)


def _build_strategy_fn(name: str) -> Callable[..., np.ndarray]:
    if name == "two_seek":
        return _two_seek_get_frames_by_timestamps
    if name == "single_seek":
        return _single_seek_get_frames_by_timestamps
    raise ValueError(f"Unknown strategy: {name}")


def _patch_fetch_strategy(strategy_fn: Callable[..., np.ndarray]) -> None:
    import starVLA.dataloader.gr00t_lerobot.datasets as ds_mod
    import starVLA.dataloader.gr00t_lerobot.video as video_mod

    video_mod.get_frames_by_timestamps = strategy_fn
    ds_mod.get_frames_by_timestamps = strategy_fn


def _build_datamodule(data_cfg: dict[str, Any]):
    from latent_action_model.data_loader.lerobot_datamodule import LeRobotDataModule

    required = ["data_root_dir", "data_mix", "num_frames", "frame_dt_sec"]
    missing = [k for k in required if k not in data_cfg]
    if missing:
        raise ValueError(f"Missing required keys in data cfg: {missing}")
    return LeRobotDataModule(**data_cfg)


def _summarize(batch_times: list[float], warmup: int) -> dict[str, float]:
    if not batch_times:
        raise ValueError("No batch timings available")

    used = batch_times[warmup:] if warmup < len(batch_times) else batch_times
    mean_s = statistics.mean(used)
    p50_s = statistics.median(used)
    p90_s = float(np.percentile(np.asarray(used), 90))
    return {
        "num_batches": len(batch_times),
        "num_used": len(used),
        "warmup": warmup,
        "mean_s": mean_s,
        "p50_s": p50_s,
        "p90_s": p90_s,
        "ips_mean": 1.0 / mean_s if mean_s > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LeRobot dataloader video fetch strategy.")
    parser.add_argument("--config", type=Path, required=True, help="Path to yaml config")
    parser.add_argument("--data-mix", type=str, default="droid", help="Override data_mix")
    parser.add_argument("--strategy", type=str, choices=["two_seek", "single_seek"], required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    cfg = _load_data_cfg(args.config)
    cfg["data_mix"] = args.data_mix
    cfg["video_backend"] = "pyav"
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.prefetch_factor is not None:
        cfg["prefetch_factor"] = args.prefetch_factor

    strategy_fn = _build_strategy_fn(args.strategy)
    _patch_fetch_strategy(strategy_fn)

    print(
        f"[bench] strategy={args.strategy} mix={cfg['data_mix']} "
        f"batch_size={cfg.get('batch_size')} num_workers={cfg.get('num_workers')} "
        f"prefetch_factor={cfg.get('prefetch_factor')} steps={args.steps} warmup={args.warmup}"
    )

    dm = _build_datamodule(cfg)
    dm.setup("fit")
    loader = dm.train_dataloader()
    it = iter(loader)

    batch_times: list[float] = []
    t_begin = time.perf_counter()
    for i in range(args.steps):
        t0 = time.perf_counter()
        _ = next(it)
        dt = time.perf_counter() - t0
        batch_times.append(dt)
        if (i + 1) % 10 == 0:
            print(f"[bench] step={i+1}/{args.steps} last_batch_s={dt:.4f}")

    total_s = time.perf_counter() - t_begin
    summary = _summarize(batch_times, args.warmup)
    summary["total_wall_s"] = total_s
    print("[bench] summary_json=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
