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
from PIL import Image
from tqdm import tqdm

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


def _to_emb_list(emb: Any, batch_size: int) -> list[int | None]:
    if isinstance(emb, torch.Tensor):
        if emb.ndim == 0:
            return [int(emb.item())] * batch_size
        emb_flat = emb.detach().cpu().reshape(-1).tolist()
        return [int(v) for v in emb_flat[:batch_size]]
    if isinstance(emb, (list, tuple)):
        out: list[int | None] = []
        for v in list(emb)[:batch_size]:
            try:
                out.append(int(v))
            except Exception:
                out.append(None)
        return out
    return [None] * batch_size


def _grayscale_u8(frame_u8: torch.Tensor) -> torch.Tensor:
    # frame_u8: [H,W,C] uint8
    if frame_u8.ndim != 3 or frame_u8.shape[-1] != 3:
        raise ValueError(f"Expected frame [H,W,3], got {tuple(frame_u8.shape)}")
    f = frame_u8.to(torch.float32)
    gray = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return gray.round().clamp_(0, 255).to(torch.uint8)


def _entropy_u8(gray_u8: torch.Tensor) -> float:
    hist = torch.bincount(gray_u8.reshape(-1), minlength=256).to(torch.float32)
    p = hist / hist.sum().clamp_min(1.0)
    nz = p > 0
    return float((-(p[nz] * torch.log2(p[nz]))).sum().item())


def _roughness_u8(gray_u8: torch.Tensor) -> float:
    g = gray_u8.to(torch.float32)
    if g.shape[0] < 2 or g.shape[1] < 2:
        return 0.0
    dx = torch.abs(g[:, 1:] - g[:, :-1]).mean()
    dy = torch.abs(g[1:, :] - g[:-1, :]).mean()
    return float(((dx + dy) * 0.5).item())


def _quantile_u8(gray_u8: torch.Tensor, q: float) -> float:
    # q in [0, 1]
    return float(torch.quantile(gray_u8.to(torch.float32), q).item())


def _save_anomaly_frame(
    frame_u8: torch.Tensor,
    out_dir: Path,
    *,
    step: int,
    epoch: int,
    sample_idx: int,
    frame_idx: int,
    emb: str,
    reason_tag: str,
    save_state: dict[str, int],
    max_save: int,
) -> None:
    if save_state["count"] >= max_save:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = (
        f"step{step:07d}_ep{epoch:03d}_sample{sample_idx:03d}_frame{frame_idx:02d}"
        f"_emb{emb}_{reason_tag}.png"
    )
    img_path = out_dir / file_name
    arr = frame_u8.detach().cpu().numpy()
    Image.fromarray(arr).save(img_path)
    save_state["count"] += 1


def _check_videos_for_anomalies(
    videos: Any,
    embodiment_ids: Any,
    *,
    step: int,
    epoch: int,
    black_pixel_threshold: int,
    white_pixel_threshold: int,
    dark_ratio_threshold: float,
    bright_ratio_threshold: float,
    low_dynamic_range_threshold: float,
    low_std_threshold: float,
    clipping_ratio_threshold: float,
    channel_dominance_threshold: float,
    duplicate_frame_ratio_threshold: float,
    temporal_jump_threshold: float,
    entropy_low_threshold: float,
    entropy_high_threshold: float,
    roughness_high_threshold: float,
    save_anomaly_dir: Path | None,
    save_state: dict[str, int],
    max_save_anomalies: int,
) -> int:
    """
    Check every sample in current batch for frame-value anomalies.

    Returns:
        int: number of anomalous samples in this batch.
    """
    if not isinstance(videos, torch.Tensor):
        print(
            f"[probe][anomaly] step={step} epoch={epoch} "
            f"videos is not Tensor: type={type(videos)}"
        )
        return 1
    if videos.ndim != 5:
        print(
            f"[probe][anomaly] step={step} epoch={epoch} "
            f"videos has invalid ndim={videos.ndim}, shape={tuple(videos.shape)}"
        )
        return int(videos.shape[0]) if videos.ndim > 0 else 1

    B = int(videos.shape[0])
    emb_list = _to_emb_list(embodiment_ids, B)
    anomalies = 0

    # Expected in this pipeline: uint8 [B,T,H,W,C]
    expected_uint8 = videos.dtype == torch.uint8
    if not expected_uint8:
        print(
            f"[probe][warn] step={step} epoch={epoch} "
            f"videos dtype={videos.dtype} (expected torch.uint8)"
        )

    for i in range(B):
        sample = videos[i]
        emb = emb_list[i] if i < len(emb_list) else None
        emb_str = "unknown" if emb is None else str(emb)

        # finite check (for float tensors only)
        if torch.is_floating_point(sample):
            finite_mask = torch.isfinite(sample)
            if not bool(finite_mask.all()):
                non_finite = int((~finite_mask).sum().item())
                print(
                    f"[probe][anomaly] step={step} epoch={epoch} sample={i} emb={emb_str} "
                    f"non_finite={non_finite}"
                )
                anomalies += 1
                continue
            valid = sample
        else:
            valid = sample

        # min/max range check
        mn = float(valid.min().item())
        mx = float(valid.max().item())
        is_anomaly = False
        reasons: list[str] = []

        if torch.is_floating_point(valid):
            # In this pipeline videos should be uint8-like values.
            if mn < -1e-3 or mx > 255.0 + 1e-3:
                is_anomaly = True
                reasons.append(f"range=[{mn:.4f},{mx:.4f}] outside [0,255]")
            # Flag suspicious normalization for easier debugging.
            if 0.0 - 1e-3 <= mn and mx <= 1.0 + 1e-3:
                is_anomaly = True
                reasons.append(f"range=[{mn:.4f},{mx:.4f}] looks normalized [0,1]")
        else:
            # Integer tensors: ensure byte-like range.
            if mn < 0 or mx > 255:
                is_anomaly = True
                reasons.append(f"integer range=[{mn:.0f},{mx:.0f}] outside [0,255]")

        if mn == mx:
            is_anomaly = True
            reasons.append(f"constant_value={mn:.4f}")

        # Image-quality checks for black / garbled frames.
        # valid expected shape: [T,H,W,C]
        if valid.ndim == 4 and valid.shape[-1] == 3:
            sample_u8 = valid.to(torch.uint8) if valid.dtype != torch.uint8 else valid
            T = int(sample_u8.shape[0])
            frame_reasons: list[str] = []
            for t in range(T):
                frame = sample_u8[t]
                # dark ratio over pixels where all channels are <= threshold
                dark_ratio = float((frame.max(dim=-1).values <= black_pixel_threshold).to(torch.float32).mean().item())
                bright_ratio = float((frame.min(dim=-1).values >= white_pixel_threshold).to(torch.float32).mean().item())
                frame_max = int(frame.max().item())
                frame_min = int(frame.min().item())
                gray = _grayscale_u8(frame)
                g_f = gray.to(torch.float32)
                entropy = _entropy_u8(gray)
                rough = _roughness_u8(gray)
                g_std = float(g_f.std().item())
                q01 = _quantile_u8(gray, 0.01)
                q99 = _quantile_u8(gray, 0.99)
                dyn_range = q99 - q01
                clip_ratio = float(((gray <= 1) | (gray >= 254)).to(torch.float32).mean().item())
                ch_mean = frame.to(torch.float32).mean(dim=(0, 1))
                ch_sum = float(ch_mean.sum().item())
                ch_dom = float(ch_mean.max().item() / (ch_sum + 1e-6))

                fr_flags: list[str] = []
                if frame_max <= black_pixel_threshold:
                    fr_flags.append(f"all_black(max={frame_max})")
                if dark_ratio >= dark_ratio_threshold:
                    fr_flags.append(f"near_black(dark_ratio={dark_ratio:.3f})")
                if frame_min >= white_pixel_threshold:
                    fr_flags.append(f"all_white(min={frame_min})")
                if bright_ratio >= bright_ratio_threshold:
                    fr_flags.append(f"near_white(bright_ratio={bright_ratio:.3f})")
                if dyn_range <= low_dynamic_range_threshold:
                    fr_flags.append(f"low_dynamic_range({dyn_range:.2f})")
                if g_std <= low_std_threshold:
                    fr_flags.append(f"low_std({g_std:.2f})")
                if clip_ratio >= clipping_ratio_threshold:
                    fr_flags.append(f"heavy_clipping({clip_ratio:.3f})")
                if ch_dom >= channel_dominance_threshold:
                    fr_flags.append(f"channel_dominance({ch_dom:.3f})")
                if entropy <= entropy_low_threshold:
                    fr_flags.append(f"low_entropy({entropy:.3f})")
                if entropy >= entropy_high_threshold and rough >= roughness_high_threshold:
                    fr_flags.append(f"possible_garble(ent={entropy:.3f},rough={rough:.2f})")
                if t > 0:
                    prev = sample_u8[t - 1]
                    # exact-repeat ratio: per-pixel max channel diff == 0
                    exact_same_ratio = float(
                        (torch.abs(frame.to(torch.int16) - prev.to(torch.int16)).max(dim=-1).values == 0)
                        .to(torch.float32)
                        .mean()
                        .item()
                    )
                    # mean absolute difference as a coarse temporal jump indicator
                    temporal_mad = float(torch.abs(frame.to(torch.float32) - prev.to(torch.float32)).mean().item())
                    if exact_same_ratio >= duplicate_frame_ratio_threshold:
                        fr_flags.append(f"duplicate_frame(prev,ratio={exact_same_ratio:.3f})")
                    if temporal_mad >= temporal_jump_threshold:
                        fr_flags.append(f"temporal_jump(prev,mad={temporal_mad:.2f})")

                if fr_flags:
                    is_anomaly = True
                    flag_str = ",".join(fr_flags)
                    frame_reasons.append(
                        "frame"
                        f"{t}:min={frame_min},max={frame_max},dark={dark_ratio:.3f},bright={bright_ratio:.3f},"
                        f"std={g_std:.2f},dyn={dyn_range:.2f},clip={clip_ratio:.3f},chdom={ch_dom:.3f},"
                        f"ent={entropy:.3f},rough={rough:.2f},flags={flag_str}"
                    )
                    if save_anomaly_dir is not None:
                        reason_tag = "black" if any("black" in f for f in fr_flags) else "img_abnormal"
                        _save_anomaly_frame(
                            frame,
                            save_anomaly_dir,
                            step=step,
                            epoch=epoch,
                            sample_idx=i,
                            frame_idx=t,
                            emb=emb_str,
                            reason_tag=reason_tag,
                            save_state=save_state,
                            max_save=max_save_anomalies,
                        )

            if frame_reasons:
                reasons.extend(frame_reasons)

        if is_anomaly:
            anomalies += 1
            print(
                f"[probe][anomaly] step={step} epoch={epoch} sample={i} emb={emb_str} "
                f"dtype={valid.dtype} shape={tuple(valid.shape)} "
                + "; ".join(reasons)
            )

    return anomalies


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
        default=None,
        help="Max successful batches to run; if unset, defaults to one dataset epoch (len(train_dataloader)). Use -1 for infinite.",
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
    parser.add_argument(
        "--black-pixel-threshold",
        type=int,
        default=5,
        help="Pixel value <= this threshold is treated as black for dark-ratio checks.",
    )
    parser.add_argument(
        "--white-pixel-threshold",
        type=int,
        default=250,
        help="Pixel value >= this threshold is treated as white for bright-ratio checks.",
    )
    parser.add_argument(
        "--dark-ratio-threshold",
        type=float,
        default=0.90,
        help="Frame is suspicious if dark pixel ratio >= this threshold.",
    )
    parser.add_argument(
        "--bright-ratio-threshold",
        type=float,
        default=0.90,
        help="Frame is suspicious if bright pixel ratio >= this threshold.",
    )
    parser.add_argument(
        "--low-dynamic-range-threshold",
        type=float,
        default=14.0,
        help="Frame is suspicious if (q99 - q01) <= this threshold.",
    )
    parser.add_argument(
        "--low-std-threshold",
        type=float,
        default=6.0,
        help="Frame is suspicious if grayscale std <= this threshold.",
    )
    parser.add_argument(
        "--clipping-ratio-threshold",
        type=float,
        default=0.90,
        help="Frame is suspicious if ratio(gray<=1 or gray>=254) >= this threshold.",
    )
    parser.add_argument(
        "--channel-dominance-threshold",
        type=float,
        default=0.80,
        help="Frame is suspicious if one channel mean占比 >= this threshold.",
    )
    parser.add_argument(
        "--duplicate-frame-ratio-threshold",
        type=float,
        default=0.995,
        help="Frame is suspicious if exact pixel match ratio vs previous frame >= this threshold.",
    )
    parser.add_argument(
        "--temporal-jump-threshold",
        type=float,
        default=90.0,
        help="Frame is suspicious if MAD vs previous frame >= this threshold.",
    )
    parser.add_argument(
        "--entropy-low-threshold",
        type=float,
        default=2.0,
        help="Frame is suspicious if grayscale entropy <= this threshold.",
    )
    parser.add_argument(
        "--entropy-high-threshold",
        type=float,
        default=7.6,
        help="Used with roughness threshold to detect possible garbled/noise frames.",
    )
    parser.add_argument(
        "--roughness-high-threshold",
        type=float,
        default=70.0,
        help="Used with high entropy to detect potential garbled/noise frames.",
    )
    parser.add_argument(
        "--save-anomaly-dir",
        type=Path,
        default=REPO_ROOT / "latent_action_model" / "artifacts" / "probe_anomalies",
        help="Directory to save suspicious frames for manual inspection.",
    )
    parser.add_argument(
        "--max-save-anomalies",
        type=int,
        default=200,
        help="Max number of suspicious frames to save in one run.",
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
    loader_len = len(loader)

    print(
        "[probe] dataloader ready: "
        f"batch_size={loader.batch_size}, num_workers={loader.num_workers}, "
        f"prefetch_factor={loader.prefetch_factor}, pin_memory={loader.pin_memory}, "
        f"loader_len={loader_len}"
    )

    resolved_max_steps = args.max_steps
    if resolved_max_steps is None:
        resolved_max_steps = loader_len
        print(
            "[probe] max-steps not set; defaulting to one dataset epoch "
            f"(max_steps={resolved_max_steps})."
        )
    elif resolved_max_steps == -1:
        print("[probe] max_steps=-1, running in infinite mode.")
    elif resolved_max_steps <= 0:
        raise ValueError(f"--max-steps must be positive, -1, or unset. Got: {resolved_max_steps}")

    pbar_total = None if resolved_max_steps == -1 else int(resolved_max_steps)
    pbar = tqdm(total=pbar_total, desc="probe", unit="batch")

    step = 0
    epoch = 0
    anomaly_samples = 0
    anomaly_batches = 0
    save_state = {"count": 0}
    t0 = time.time()
    last_log_t = t0
    it = iter(loader)

    try:
        while True:
            if resolved_max_steps != -1 and step >= int(resolved_max_steps):
                print(f"[probe] reached max steps={resolved_max_steps}, exiting.")
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
            pbar.update(1)
            batch_anomalies = _check_videos_for_anomalies(
                batch.get("videos"),
                batch.get("embodiment_ids"),
                step=step,
                epoch=epoch,
                black_pixel_threshold=args.black_pixel_threshold,
                white_pixel_threshold=args.white_pixel_threshold,
                dark_ratio_threshold=args.dark_ratio_threshold,
                bright_ratio_threshold=args.bright_ratio_threshold,
                low_dynamic_range_threshold=args.low_dynamic_range_threshold,
                low_std_threshold=args.low_std_threshold,
                clipping_ratio_threshold=args.clipping_ratio_threshold,
                channel_dominance_threshold=args.channel_dominance_threshold,
                duplicate_frame_ratio_threshold=args.duplicate_frame_ratio_threshold,
                temporal_jump_threshold=args.temporal_jump_threshold,
                entropy_low_threshold=args.entropy_low_threshold,
                entropy_high_threshold=args.entropy_high_threshold,
                roughness_high_threshold=args.roughness_high_threshold,
                save_anomaly_dir=args.save_anomaly_dir,
                save_state=save_state,
                max_save_anomalies=args.max_save_anomalies,
            )
            if batch_anomalies > 0:
                anomaly_batches += 1
                anomaly_samples += batch_anomalies

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
                    f"state_mask={_shape_str(state_mask)} emb={_shape_str(emb)} "
                    f"anomaly_batches={anomaly_batches} anomaly_samples={anomaly_samples} "
                    f"saved_anomaly_frames={save_state['count']}"
                )
    finally:
        pbar.close()


if __name__ == "__main__":
    main()
