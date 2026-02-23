#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset as HFDataset


def _ensure_starvla_on_path() -> None:
    if importlib.util.find_spec("starVLA") is not None:
        return
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    if importlib.util.find_spec("starVLA") is None:
        raise ModuleNotFoundError(
            "Cannot import 'starVLA'. Run from repo root or add it to PYTHONPATH."
        )


_ensure_starvla_on_path()

from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES


DATA_MIX_ALIASES = {
    "libero_4in1": "libero",
    "libero4in1": "libero",
    "libero_all": "libero",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify each dataset in a named mixture as delta/absolute using only ACF-based "
            "time-series statistics."
        )
    )
    parser.add_argument("--data-root-dir", type=Path, required=True)
    parser.add_argument("--data-mix", type=str, required=True)
    parser.add_argument("--data-mix-fallback", type=str, default=None)
    parser.add_argument("--max-episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=80000)
    parser.add_argument("--max-lag", type=int, default=20)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser


def _resolve_mix_name(data_mix: str, fallback_mix: str | None) -> str:
    if data_mix in DATASET_NAMED_MIXTURES:
        return data_mix

    if fallback_mix and fallback_mix in DATASET_NAMED_MIXTURES:
        print(f"[INFO] data_mix `{data_mix}` not found; falling back to `{fallback_mix}`.")
        return fallback_mix

    alias = DATA_MIX_ALIASES.get(data_mix)
    if alias in DATASET_NAMED_MIXTURES:
        print(f"[INFO] data_mix `{data_mix}` not found; using alias `{alias}`.")
        return alias

    raise KeyError(
        f"Unknown data_mix `{data_mix}`. Available mixes: {sorted(DATASET_NAMED_MIXTURES.keys())}"
    )


def _resolve_mixture_entries(mix_name: str) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    entries: list[dict[str, Any]] = []
    for dataset_name, weight, robot_type in DATASET_NAMED_MIXTURES[mix_name]:
        key = (dataset_name, robot_type)
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            {
                "dataset_name": dataset_name,
                "weight": float(weight),
                "robot_type": robot_type,
            }
        )
    return entries


def _numeric_path_sort_key(path: Path) -> tuple[int, int]:
    nums = [int(x) for x in re.findall(r"\d+", str(path))]
    if len(nums) >= 2:
        return nums[-2], nums[-1]
    if len(nums) == 1:
        return nums[0], 0
    return 0, 0


def _collect_parquet_files(dataset_root: Path, rel_glob: str) -> list[Path]:
    files = sorted(dataset_root.glob(rel_glob), key=_numeric_path_sort_key)
    return [p for p in files if p.exists()]


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_modality_slices(modality_json: dict[str, Any], modality: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    modality_fields = modality_json.get(modality, {})
    if not isinstance(modality_fields, dict):
        return out
    for field_name, field_meta in modality_fields.items():
        if not isinstance(field_meta, dict):
            continue
        if "start" not in field_meta or "end" not in field_meta:
            continue
        out.append(
            {
                "field_name": field_name,
                "source_key": field_meta.get("original_key") or ("action" if modality == "action" else "observation.state"),
                "start": int(field_meta["start"]),
                "end": int(field_meta["end"]),
            }
        )
    return out


def _to_1d_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def _stack_slice_column(column_values: Any, start: int, end: int) -> np.ndarray:
    rows = [_to_1d_array(v)[start:end] for v in column_values]
    if not rows:
        return np.empty((0, max(end - start, 0)), dtype=np.float32)
    return np.stack(rows, axis=0)


def _build_episode_matrix(df, field_slices: list[dict[str, Any]]) -> np.ndarray | None:
    parts: list[np.ndarray] = []
    for field in field_slices:
        source_key = field["source_key"]
        if source_key not in df.columns:
            continue
        start = field["start"]
        end = field["end"]
        if end <= start:
            continue
        part = _stack_slice_column(df[source_key].to_numpy(), start, end)
        if part.size == 0:
            continue
        parts.append(part)
    if not parts:
        return None
    return np.concatenate(parts, axis=1)


def _is_low_information_dim(x: np.ndarray) -> bool:
    if x.size < 8:
        return True
    if not np.isfinite(x).all():
        return True
    if np.std(x) < 1e-6:
        return True
    if np.unique(np.round(x, 6)).size <= 3:
        return True
    return False


def _acf_curve_1d(x: np.ndarray, max_lag: int) -> np.ndarray | None:
    if x.size <= max_lag + 2:
        return None
    x = x.astype(np.float64)
    x = x - np.mean(x)
    var = np.var(x)
    if not np.isfinite(var) or var < 1e-10:
        return None

    curve = np.empty(max_lag, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        curve[lag - 1] = np.dot(x[:-lag], x[lag:]) / ((x.size - lag) * var)
    return curve


def _mean_acf_curve(sequences: list[np.ndarray], max_lag: int) -> tuple[np.ndarray | None, int]:
    curves: list[np.ndarray] = []
    used_dims = 0
    for seq in sequences:
        if seq.shape[0] <= max_lag + 2:
            continue
        for dim in range(seq.shape[1]):
            x = seq[:, dim]
            if _is_low_information_dim(x):
                continue
            curve = _acf_curve_1d(x, max_lag)
            if curve is None or not np.isfinite(curve).all():
                continue
            curves.append(curve)
            used_dims += 1
    if not curves:
        return None, 0
    return np.mean(np.stack(curves, axis=0), axis=0), used_dims


def _classify_by_acf(
    dataset_root: Path,
    max_episodes: int,
    max_steps: int,
    max_lag: int,
    margin: float,
) -> dict[str, Any]:
    modality_path = dataset_root / "meta" / "modality.json"
    if not modality_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {modality_path}")

    episode_files = _collect_parquet_files(dataset_root, "meta/episodes/*/*.parquet")
    data_files = _collect_parquet_files(dataset_root, "data/*/*.parquet")
    if not episode_files:
        raise FileNotFoundError(f"No episode parquet files under {dataset_root / 'meta' / 'episodes'}")
    if not data_files:
        raise FileNotFoundError(f"No data parquet files under {dataset_root / 'data'}")

    modality_json = _load_json(modality_path)
    action_slices = _parse_modality_slices(modality_json, "action")
    state_slices = _parse_modality_slices(modality_json, "state")
    if not action_slices:
        return {
            "status": "uncertain",
            "reason": "no_action_slices",
            "used_episodes": 0,
            "used_steps": 0,
        }
    if not state_slices:
        return {
            "status": "uncertain",
            "reason": "no_state_slices",
            "used_episodes": 0,
            "used_steps": 0,
        }

    episodes_ds = HFDataset.from_parquet([str(p) for p in episode_files])
    data_ds = HFDataset.from_parquet([str(p) for p in data_files])

    action_sequences: list[np.ndarray] = []
    state_sequences: list[np.ndarray] = []
    used_steps = 0
    used_episodes = 0

    for episode_idx in range(len(episodes_ds)):
        if used_episodes >= max_episodes or used_steps >= max_steps:
            break
        row = episodes_ds[episode_idx]
        from_idx = int(row["dataset_from_index"])
        to_idx = int(row["dataset_to_index"])
        if to_idx <= from_idx:
            continue
        remain_steps = max_steps - used_steps
        to_idx = min(to_idx, from_idx + remain_steps)
        if to_idx <= from_idx:
            break

        df = data_ds.select(range(from_idx, to_idx)).to_pandas()
        action_mat = _build_episode_matrix(df, action_slices)
        state_mat = _build_episode_matrix(df, state_slices)
        if action_mat is None or state_mat is None:
            continue
        if action_mat.shape[0] <= max_lag + 2 or state_mat.shape[0] <= max_lag + 3:
            continue

        action_sequences.append(action_mat)
        state_sequences.append(state_mat)
        used_steps += action_mat.shape[0]
        used_episodes += 1

    if not action_sequences or not state_sequences:
        return {
            "status": "uncertain",
            "reason": "insufficient_samples",
            "used_episodes": used_episodes,
            "used_steps": used_steps,
        }

    acf_action, dims_action = _mean_acf_curve(action_sequences, max_lag)
    acf_state, dims_state = _mean_acf_curve(state_sequences, max_lag)
    dstate_sequences = [np.diff(seq, axis=0) for seq in state_sequences if seq.shape[0] > 4]
    acf_dstate, dims_dstate = _mean_acf_curve(dstate_sequences, max_lag)

    if acf_action is None or acf_state is None or acf_dstate is None:
        return {
            "status": "uncertain",
            "reason": "insufficient_signal",
            "used_episodes": used_episodes,
            "used_steps": used_steps,
            "used_dims_action": dims_action,
            "used_dims_state": dims_state,
            "used_dims_dstate": dims_dstate,
        }

    dist_to_state = float(np.mean(np.abs(acf_action - acf_state)))
    dist_to_dstate = float(np.mean(np.abs(acf_action - acf_dstate)))
    score_sum = dist_to_state + dist_to_dstate + 1e-8
    confidence = float(abs(dist_to_state - dist_to_dstate) / score_sum)

    if dist_to_state + margin < dist_to_dstate:
        prediction = "absolute"
        status = "ok"
    elif dist_to_dstate + margin < dist_to_state:
        prediction = "delta"
        status = "ok"
    else:
        prediction = "uncertain"
        status = "uncertain"

    return {
        "status": status,
        "prediction": prediction,
        "confidence": confidence,
        "d_abs": dist_to_state,
        "d_delta": dist_to_dstate,
        "acf_action_lag1": float(acf_action[0]),
        "acf_state_lag1": float(acf_state[0]),
        "acf_dstate_lag1": float(acf_dstate[0]),
        "used_episodes": used_episodes,
        "used_steps": used_steps,
        "used_dims_action": dims_action,
        "used_dims_state": dims_state,
        "used_dims_dstate": dims_dstate,
    }


def _print_result_row(dataset_name: str, result: dict[str, Any]) -> None:
    status = result.get("status", "error")
    if status == "error":
        print(
            f"- {dataset_name}: status=error reason={result.get('reason', 'unknown_error')}"
        )
        return

    if status == "uncertain":
        reason = result.get("reason")
        if reason:
            print(
                f"- {dataset_name}: prediction=uncertain reason={reason} "
                f"used_episodes={result.get('used_episodes', 0)} used_steps={result.get('used_steps', 0)}"
            )
            return

    prediction = result.get("prediction", "uncertain")
    confidence = result.get("confidence")
    d_abs = result.get("d_abs")
    d_delta = result.get("d_delta")
    used_episodes = result.get("used_episodes", 0)
    used_steps = result.get("used_steps", 0)

    conf_str = f"{confidence:.4f}" if isinstance(confidence, float) else "n/a"
    d_abs_str = f"{d_abs:.4f}" if isinstance(d_abs, float) else "n/a"
    d_delta_str = f"{d_delta:.4f}" if isinstance(d_delta, float) else "n/a"
    print(
        f"- {dataset_name}: prediction={prediction} confidence={conf_str} "
        f"d_abs={d_abs_str} d_delta={d_delta_str} used_episodes={used_episodes} used_steps={used_steps}"
    )


def main() -> int:
    args = _build_parser().parse_args()
    if args.max_episodes <= 0:
        raise ValueError("--max-episodes must be > 0")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.max_lag <= 0:
        raise ValueError("--max-lag must be > 0")
    if args.margin < 0:
        raise ValueError("--margin must be >= 0")

    resolved_mix = _resolve_mix_name(args.data_mix, args.data_mix_fallback)
    entries = _resolve_mixture_entries(resolved_mix)
    if not entries:
        raise RuntimeError(f"No datasets found in mix `{resolved_mix}`.")

    print(f"[INFO] data_root_dir={args.data_root_dir}")
    print(f"[INFO] data_mix={args.data_mix} resolved_mix={resolved_mix} datasets={len(entries)}")

    all_results: list[dict[str, Any]] = []
    any_error = False
    for entry in entries:
        dataset_name = entry["dataset_name"]
        dataset_root = args.data_root_dir / dataset_name
        result: dict[str, Any]

        if not dataset_root.exists():
            result = {
                "status": "error",
                "reason": "missing_dataset_dir",
                "dataset_name": dataset_name,
                "dataset_root": str(dataset_root),
            }
        else:
            try:
                result = _classify_by_acf(
                    dataset_root=dataset_root,
                    max_episodes=args.max_episodes,
                    max_steps=args.max_steps,
                    max_lag=args.max_lag,
                    margin=args.margin,
                )
                result["dataset_name"] = dataset_name
                result["dataset_root"] = str(dataset_root)
            except Exception as exc:
                result = {
                    "status": "error",
                    "reason": type(exc).__name__,
                    "error_message": str(exc),
                    "dataset_name": dataset_name,
                    "dataset_root": str(dataset_root),
                }

        all_results.append(result)
        _print_result_row(dataset_name, result)

        if result.get("status") == "error":
            any_error = True
            if args.strict:
                break

    summary = {
        "data_root_dir": str(args.data_root_dir),
        "data_mix": args.data_mix,
        "resolved_mix": resolved_mix,
        "max_episodes": args.max_episodes,
        "max_steps": args.max_steps,
        "max_lag": args.max_lag,
        "margin": args.margin,
        "results": all_results,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved JSON results to {args.output_json}")

    if args.strict and any_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
