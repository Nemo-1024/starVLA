#!/usr/bin/env python3
"""WebSocket policy server that replays actions from a LeRobot dataset trajectory.

This server is API-compatible with `ModelClient` in examples/LIBERO/eval_files/model2libero_interface.py,
returning `{"data": {"normalized_actions": ...}}` through WebsocketPolicyServer.
"""

import argparse
import json
import logging
import socket
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def _pick_unnorm_key(norm_stats: dict, requested_key: Optional[str]) -> str:
    if requested_key is None:
        if len(norm_stats) != 1:
            raise ValueError(
                "Checkpoint has multiple dataset stats. "
                f"Please provide --unnorm-key from: {list(norm_stats.keys())}"
            )
        return next(iter(norm_stats.keys()))
    if requested_key not in norm_stats:
        raise KeyError(
            f"Unknown --unnorm-key `{requested_key}`. Available keys: {list(norm_stats.keys())}"
        )
    return requested_key


class DatasetTrajectoryReplayPolicy:
    def __init__(
        self,
        dataset_root: str | Path,
        episode_index: int,
        action_chunk_size: int,
        loop: bool,
        gripper_convention: str,
        normalize_source: str,
        ckpt_path: Optional[str],
        unnorm_key: Optional[str],
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.episode_index = int(episode_index)
        self.action_chunk_size = int(action_chunk_size)
        self.loop = bool(loop)
        self.gripper_convention = gripper_convention
        self.normalize_source = normalize_source
        self.cursor = 0

        if self.action_chunk_size <= 0:
            raise ValueError(f"action_chunk_size must be > 0, got {self.action_chunk_size}")

        self._normalizer = self._build_normalizer(
            normalize_source=normalize_source,
            ckpt_path=ckpt_path,
            unnorm_key=unnorm_key,
        )

        raw_actions = self._load_episode_actions(self.episode_index)
        if raw_actions.ndim != 2 or raw_actions.shape[1] < 7:
            raise ValueError(
                "Expected episode actions with shape [T, >=7], got "
                f"{tuple(raw_actions.shape)} from {self.dataset_root}"
            )
        self.raw_actions = raw_actions[:, :7].astype(np.float32, copy=False)
        self.normalized_actions = self._to_client_normalized_actions(self.raw_actions)

        logging.info(
            "Loaded replay trajectory: dataset=%s, episode=%d, steps=%d, chunk=%d, loop=%s",
            self.dataset_root,
            self.episode_index,
            self.raw_actions.shape[0],
            self.action_chunk_size,
            self.loop,
        )

    def _load_stats_gr00t_action_minmax(self) -> tuple[np.ndarray, np.ndarray]:
        stats_path = self.dataset_root / "meta" / "stats_gr00t.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing stats file: {stats_path}")
        with open(stats_path, "r") as f:
            stats = json.load(f)
        if "action" not in stats:
            raise KeyError(f"`action` not found in {stats_path}")
        action_stats = stats["action"]
        low = np.asarray(action_stats["min"], dtype=np.float32)
        high = np.asarray(action_stats["max"], dtype=np.float32)
        if low.shape[0] < 7 or high.shape[0] < 7:
            raise ValueError(
                f"stats_gr00t action dim is smaller than 7: min={low.shape}, max={high.shape}, path={stats_path}"
            )
        return low[:7], high[:7]

    def _build_normalizer(
        self,
        normalize_source: str,
        ckpt_path: Optional[str],
        unnorm_key: Optional[str],
    ):
        if normalize_source == "stats_gr00t":
            low, high = self._load_stats_gr00t_action_minmax()
            logging.info("Using stats_gr00t min/max for action normalization: %s", self.dataset_root / "meta" / "stats_gr00t.json")
            return {
                "low": low,
                "high": high,
                "source": "stats_gr00t",
            }

        if normalize_source == "ckpt":
            if not ckpt_path:
                raise ValueError("--normalize-source ckpt requires --ckpt-path")
            from starVLA.model.tools import read_mode_config

            _, norm_stats = read_mode_config(ckpt_path)
            picked_key = _pick_unnorm_key(norm_stats, unnorm_key)
            action_stats = norm_stats[picked_key]["action"]
            low = np.asarray(action_stats["min"], dtype=np.float32)
            high = np.asarray(action_stats["max"], dtype=np.float32)

            if low.shape[0] < 7 or high.shape[0] < 7:
                raise ValueError(
                    f"Action stats dim is smaller than 7: min={low.shape}, max={high.shape}, unnorm_key={picked_key}"
                )

            logging.info("Using checkpoint min/max for action normalization: unnorm_key=%s", picked_key)
            return {
                "low": low[:7],
                "high": high[:7],
                "source": f"ckpt:{picked_key}",
            }

        raise ValueError(f"Unsupported --normalize-source: {normalize_source}")

    def _read_episode_table(self) -> pd.DataFrame:
        episode_files = sorted((self.dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
        if not episode_files:
            raise FileNotFoundError(
                f"No episode metadata found under {self.dataset_root / 'meta' / 'episodes'}"
            )
        tables = [pd.read_parquet(p) for p in episode_files]
        ep_df = pd.concat(tables, axis=0, ignore_index=True)
        if "episode_index" not in ep_df.columns:
            raise KeyError("Episode metadata missing required column: episode_index")
        return ep_df

    def _load_episode_actions(self, episode_index: int) -> np.ndarray:
        ep_df = self._read_episode_table()
        row = ep_df.loc[ep_df["episode_index"] == episode_index]
        if row.empty:
            available = (int(ep_df["episode_index"].min()), int(ep_df["episode_index"].max()))
            raise IndexError(
                f"episode_index={episode_index} not found in {self.dataset_root}. "
                f"Available range: [{available[0]}, {available[1]}]"
            )
        row = row.iloc[0]

        data_chunk = int(row["data/chunk_index"])
        data_file = int(row["data/file_index"])
        data_from = int(row["dataset_from_index"])
        data_to = int(row["dataset_to_index"])

        parquet_path = self.dataset_root / "data" / f"chunk-{data_chunk:03d}" / f"file-{data_file:03d}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Data file not found: {parquet_path}")

        df = pd.read_parquet(parquet_path, columns=["action", "episode_index"])  # local row range in this file

        candidate = df.iloc[data_from:data_to]
        if len(candidate) > 0 and np.all(candidate["episode_index"].to_numpy() == episode_index):
            return np.stack(candidate["action"].to_numpy())

        # Fallback for unusual index bookkeeping: filter by episode in designated file.
        filtered = df.loc[df["episode_index"] == episode_index]
        if len(filtered) > 0:
            return np.stack(filtered["action"].to_numpy())

        # Last fallback: search all data files.
        for p in sorted((self.dataset_root / "data").glob("chunk-*/*.parquet")):
            part = pd.read_parquet(p, columns=["action", "episode_index"])
            part = part.loc[part["episode_index"] == episode_index]
            if len(part) > 0:
                return np.stack(part["action"].to_numpy())

        raise RuntimeError(f"Failed to locate actions for episode_index={episode_index} in {self.dataset_root}")

    def _map_gripper_to_open_prob(self, raw_gripper: np.ndarray) -> np.ndarray:
        if self.gripper_convention == "neg_open":
            open_flag = raw_gripper < 0
        elif self.gripper_convention == "pos_open":
            open_flag = raw_gripper > 0
        elif self.gripper_convention == "zero_one_open":
            open_flag = raw_gripper > 0.5
        else:
            raise ValueError(f"Unsupported gripper convention: {self.gripper_convention}")
        return open_flag.astype(np.float32)

    def _normalize_minmax(self, raw_actions: np.ndarray) -> np.ndarray:
        low = self._normalizer["low"]
        high = self._normalizer["high"]
        denom = np.maximum(high - low, 1e-6)
        out = 2.0 * (raw_actions - low) / denom - 1.0
        out = out.astype(np.float32, copy=False)
        out[:, :6] = np.clip(out[:, :6], -1.0, 1.0)
        return out

    def _to_client_normalized_actions(self, raw_actions: np.ndarray) -> np.ndarray:
        normalized = self._normalize_minmax(raw_actions)
        # Client expects `normalized_actions[..., 6]` thresholded into open_gripper in {0, 1}.
        normalized[:, 6] = self._map_gripper_to_open_prob(raw_actions[:, 6])
        return normalized.astype(np.float32, copy=False)

    def _next_chunk(self) -> np.ndarray:
        T = self.normalized_actions.shape[0]
        K = self.action_chunk_size
        if T == 0:
            raise RuntimeError("Trajectory has zero actions")

        if self.loop:
            idx = (np.arange(self.cursor, self.cursor + K) % T).astype(np.int64)
            chunk = self.normalized_actions[idx]
            self.cursor = int((self.cursor + K) % T)
            return chunk

        start = self.cursor
        end = min(start + K, T)
        chunk = self.normalized_actions[start:end]
        self.cursor = end
        if chunk.shape[0] < K:
            pad = np.repeat(self.normalized_actions[-1:], repeats=K - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        return chunk

    def predict_action(self, **_: dict) -> dict:
        chunk = self._next_chunk()
        return {
            "normalized_actions": np.expand_dims(chunk, axis=0),  # [1, chunk, 7]
        }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay LeRobot trajectory as a fake WebSocket policy server.")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to dataset root (contains meta/ and data/).")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index to replay.")
    parser.add_argument("--port", type=int, default=5694)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--idle-timeout", type=int, default=-1, help="Server idle timeout in seconds, -1 to disable.")
    parser.add_argument("--action-chunk-size", type=int, default=8, help="Chunk length returned per predict call.")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True, help="Loop trajectory when reaching the end.")
    parser.add_argument(
        "--normalize-source",
        type=str,
        default="stats_gr00t",
        choices=["stats_gr00t", "ckpt"],
        help="Normalization source for action min/max.",
    )
    parser.add_argument(
        "--gripper-convention",
        type=str,
        default="neg_open",
        choices=["neg_open", "pos_open", "zero_one_open"],
        help="How to interpret raw dataset gripper action for open/close.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help=(
            "Optional .pt checkpoint path. If provided, server inverses the checkpoint action normalization "
            "for first 6 dims so client unnormalize recovers near-raw actions."
        ),
    )
    parser.add_argument("--unnorm-key", type=str, default=None, help="Optional dataset key for checkpoint norm stats.")
    return parser


def main(args) -> None:
    policy = DatasetTrajectoryReplayPolicy(
        dataset_root=args.dataset_root,
        episode_index=args.episode_index,
        action_chunk_size=args.action_chunk_size,
        loop=args.loop,
        gripper_convention=args.gripper_convention,
        normalize_source=args.normalize_source,
        ckpt_path=args.ckpt_path,
        unnorm_key=args.unnorm_key,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    metadata = {
        "server_type": "dataset_trajectory_replay",
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "episode_index": int(args.episode_index),
        "action_chunk_size": int(args.action_chunk_size),
        "loop": bool(args.loop),
        "gripper_convention": args.gripper_convention,
        "normalize_source": args.normalize_source,
        "trajectory_length": int(policy.raw_actions.shape[0]),
    }
    logging.info("Creating replay server (host=%s, ip=%s) with metadata=%s", hostname, local_ip, json.dumps(metadata))

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
    )
    logging.info("Replay policy server running at ws://%s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    main(parser.parse_args())
