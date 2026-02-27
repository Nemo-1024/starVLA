#!/usr/bin/env python3
"""Visualize EgoDex sample with keyframes aligned to 14-dim qpos/action curves.

Output figure layout:
- Top: keyframe contact sheet sampled from the video.
- Bottom: 14 subplots (dims 0..13). Each subplot overlays qpos and action curves,
  with vertical lines at keyframe-aligned timesteps.

This helps quickly inspect whether dims [0..6] and [7..13] correspond to left/right hands.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot EgoDex qpos/actions with video keyframes.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Task directory, e.g. /path/to/EgoDex/charge_uncharge_device",
    )
    parser.add_argument("--sample-id", type=str, required=True, help="Sample id, e.g. 0_0")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Default: <dataset-root>/vis_<sample-id>.png",
    )
    parser.add_argument("--num-keyframes", type=int, default=6, help="Number of keyframes in contact sheet.")
    parser.add_argument(
        "--sheet-height",
        type=int,
        default=180,
        help="Per-frame height (pixels) in top contact sheet.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    return parser.parse_args()


def _load_tensor(path: Path) -> np.ndarray:
    arr = torch.load(path, map_location="cpu")
    if not isinstance(arr, torch.Tensor):
        raise TypeError(f"Expected tensor at {path}, got {type(arr)}")
    out = arr.detach().cpu().numpy()
    if out.ndim != 2 or out.shape[1] != 14:
        raise ValueError(f"Expected shape [T,14] at {path}, got {out.shape}")
    return out


def _read_video_frames(video_path: Path, num_keyframes: int) -> tuple[list[np.ndarray], list[int], float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        raise RuntimeError(f"Invalid frame count from video: {video_path}")

    frame_indices = np.linspace(0, n_frames - 1, num=max(1, num_keyframes), dtype=int)
    keyframes: list[np.ndarray] = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        keyframes.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if not keyframes:
        raise RuntimeError(f"Could not decode keyframes from: {video_path}")

    used_indices = [int(x) for x in frame_indices[: len(keyframes)]]
    return keyframes, used_indices, fps, n_frames


def _build_contact_sheet(frames_rgb: list[np.ndarray], target_h: int) -> np.ndarray:
    resized = []
    for frm in frames_rgb:
        h, w = frm.shape[:2]
        new_w = max(1, int(round(w * (target_h / float(h)))))
        resized.append(cv2.resize(frm, (new_w, target_h), interpolation=cv2.INTER_AREA))
    return np.concatenate(resized, axis=1)


def main() -> None:
    args = parse_args()
    root = args.dataset_root
    sample_id = args.sample_id

    qpos_path = root / "qpos" / f"{sample_id}.pt"
    action_path = root / "actions" / f"{sample_id}.pt"
    video_path = root / "videos" / f"{sample_id}.mp4"

    for p in [qpos_path, action_path, video_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    qpos = _load_tensor(qpos_path)  # [Tq,14]
    action = _load_tensor(action_path)  # [Ta,14]

    frames, frame_indices, fps, n_frames = _read_video_frames(video_path, args.num_keyframes)
    sheet = _build_contact_sheet(frames, args.sheet_height)

    t_q = np.arange(qpos.shape[0], dtype=np.float32)
    t_a = np.arange(1, action.shape[0] + 1, dtype=np.float32)  # align action[t] with qpos[t+1]

    # Map keyframe indices to qpos timeline.
    if n_frames > 1:
        key_t = [idx * (qpos.shape[0] - 1) / (n_frames - 1) for idx in frame_indices]
    else:
        key_t = [0.0 for _ in frame_indices]

    fig = plt.figure(figsize=(19, 16), dpi=args.dpi)
    gs = fig.add_gridspec(
        nrows=8,
        ncols=2,
        height_ratios=[1.15, 1, 1, 1, 1, 1, 1, 1],
        hspace=0.28,
        wspace=0.18,
    )

    # Top contact sheet spans both columns.
    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(sheet)
    ax0.axis("off")
    frame_labels = ", ".join([f"f{idx}({idx / max(fps, 1e-6):.2f}s)" for idx in frame_indices])
    ax0.set_title(
        f"sample={sample_id} | video_frames={n_frames} @ {fps:.2f}fps | qpos={qpos.shape} action={action.shape}\\n"
        f"keyframes: {frame_labels}",
        fontsize=10,
    )

    # 14 dimensions in 7x2 layout.
    for row in range(7):
        for col in range(2):
            dim = row + (0 if col == 0 else 7)
            ax = fig.add_subplot(gs[row + 1, col])
            ax.plot(t_q, qpos[:, dim], color="#1f77b4", lw=1.4, label="qpos")
            ax.plot(t_a, action[:, dim], color="#d62728", lw=1.0, alpha=0.9, label="action")
            for kt in key_t:
                ax.axvline(kt, color="k", ls="--", lw=0.6, alpha=0.25)
            block = "block-A (0..6)" if col == 0 else "block-B (7..13)"
            ax.set_title(f"dim {dim} | {block}", fontsize=8)
            ax.grid(alpha=0.25)
            ax.set_xlim(0, max(1, qpos.shape[0] - 1))
            if row == 6:
                ax.set_xlabel("time index (qpos steps)")
            if col == 0:
                ax.set_ylabel("value")
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=7)

    if args.output is None:
        out_path = root / f"vis_{sample_id}.png"
    else:
        out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    key_t_str = ", ".join([f"{x:.2f}" for x in key_t])
    print(f"Saved: {out_path}")
    print(f"Keyframe frame indices: {frame_indices}")
    print(f"Keyframe mapped qpos steps: [{key_t_str}]")


if __name__ == "__main__":
    main()
