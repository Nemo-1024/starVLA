#!/usr/bin/env python3
"""Infinite dataloader probe for fast data-pipeline debugging.

Usage:
    python latent_action_model/scripts/infinite_dataloader_probe.py \
        --config latent_action_model/config/dino_base_ae.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # Allow running as a script: `python latent_action_model/scripts/...`
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from latent_action_model.data_loader.lerobot_datamodule import LeRobotDataModule


def _load_data_cfg(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid yaml root type: {type(cfg)}")
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        raise ValueError("Config must contain a dict field: data")
    return data_cfg


def _build_datamodule(data_cfg: dict[str, Any], args: argparse.Namespace) -> LeRobotDataModule:
    from latent_action_model.data_loader.lerobot_datamodule import LeRobotDataModule

    cfg = dict(data_cfg)

    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.prefetch_factor is not None:
        cfg["prefetch_factor"] = args.prefetch_factor
    if args.data_mix is not None:
        cfg["data_mix"] = args.data_mix
    if args.video_backend is not None:
        cfg["video_backend"] = args.video_backend

    required = ["data_root_dir", "data_mix", "num_frames", "frame_dt_sec"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required data config keys: {missing}")

    return LeRobotDataModule(**cfg)


def _shape_str(x: Any) -> str:
    if isinstance(x, torch.Tensor):
        return f"{list(x.shape)} {x.dtype}"
    return str(type(x))


def _configure_cache_dirs(cache_dir: Path, tmp_dir: Path) -> None:
    cache_dir = cache_dir.expanduser().resolve()
    tmp_dir = tmp_dir.expanduser().resolve()
    datasets_cache_dir = cache_dir / "datasets"
    hub_cache_dir = cache_dir / "hub"

    for path in (cache_dir, datasets_cache_dir, hub_cache_dir, tmp_dir):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache_dir)
    os.environ["TMPDIR"] = str(tmp_dir)

    print(
        "[probe] cache dirs: "
        f"HF_HOME={cache_dir}, HF_DATASETS_CACHE={datasets_cache_dir}, "
        f"HF_HUB_CACHE={hub_cache_dir}, TMPDIR={tmp_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Infinite dataloader iterator for debugging dataset issues.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("latent_action_model/config/dino_base_ae.yaml"),
        help="Path to training yaml config (uses only `data` section).",
    )
    parser.add_argument("--print-every", type=int, default=20, help="Print every N successful batches.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Max successful batches to run; -1 means infinite.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override data.batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override data.num_workers.")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Override data.prefetch_factor.")
    parser.add_argument("--data-mix", type=str, default=None, help="Override data.data_mix.")
    parser.add_argument("--video-backend", type=str, default=None, help="Override data.video_backend.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / ".cache" / "huggingface",
        help="Base directory for HF caches (HF_HOME/HF_DATASETS_CACHE/HF_HUB_CACHE).",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=REPO_ROOT / ".cache" / "tmp",
        help="Temp directory for large intermediate files (TMPDIR).",
    )
    args = parser.parse_args()

    if args.print_every <= 0:
        raise ValueError("--print-every must be > 0")

    torch.manual_seed(args.seed)
    config_path = args.config.resolve()
    print(f"[probe] config={config_path}")
    _configure_cache_dirs(args.cache_dir, args.tmp_dir)

    data_cfg = _load_data_cfg(config_path)
    dm = _build_datamodule(data_cfg, args)
    dm.setup("fit")
    loader = dm.train_dataloader()

    print(
        "[probe] dataloader ready: "
        f"batch_size={loader.batch_size}, num_workers={loader.num_workers}, "
        f"prefetch_factor={loader.prefetch_factor}, pin_memory={loader.pin_memory}"
    )

    step = 0
    epoch = 0
    t0 = time.time()
    last_log_t = t0
    it = iter(loader)

    while True:
        if args.max_steps >= 0 and step >= args.max_steps:
            print(f"[probe] reached max steps={args.max_steps}, exiting.")
            break

        try:
            batch = next(it)
        except StopIteration:
            epoch += 1
            if getattr(dm, "train_dataset", None) is not None and hasattr(dm.train_dataset, "mixture"):
                mixture = getattr(dm.train_dataset, "mixture", None)
                if mixture is not None and hasattr(mixture, "set_epoch"):
                    mixture.set_epoch(epoch)
            it = iter(loader)
            continue
        except Exception as exc:  # noqa: BLE001
            print("\n[probe] dataloader exception caught:")
            print(f"[probe] step={step}, epoch={epoch}, exc_type={type(exc).__name__}")
            traceback.print_exc()
            raise

        step += 1
        if step % args.print_every == 0:
            now = time.time()
            dt = now - last_log_t
            total_dt = now - t0
            ips = (args.print_every / dt) if dt > 0 else 0.0
            avg_ips = (step / total_dt) if total_dt > 0 else 0.0
            last_log_t = now

            videos = batch.get("videos")
            states = batch.get("states")
            state_mask = batch.get("state_mask")
            emb = batch.get("embodiment_ids")
            print(
                f"[probe] step={step} epoch={epoch} "
                f"ips={ips:.2f} avg_ips={avg_ips:.2f} "
                f"videos={_shape_str(videos)} states={_shape_str(states)} "
                f"state_mask={_shape_str(state_mask)} emb={_shape_str(emb)}"
            )


if __name__ == "__main__":
    main()

