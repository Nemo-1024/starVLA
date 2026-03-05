#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("/mnt/project_rlinf/jlchen/datasets/robomind_ur_1rgb")
DEFAULT_URDF_PATH = Path("/mnt/project_rlinf/jlchen/code/lerobot/ur5e.urdf")
DEFAULT_TCP_OFFSET_XYZ = [5.02498011e-06, -3.26653793e-04, 1.91330295e-01]
DEFAULT_TCP_OFFSET_RPY = [0.0, 0.0, 0.0]
DEFAULT_TCP_OFFSET_EULER_CONVENTION = "xyz"
BASE_SCRIPT = Path(__file__).resolve().parent / "add_eef_abs_action_robomind_single_dataset.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "UR-specific wrapper for adding absolute EEF action to robomind_ur_1rgb. "
            "It reuses add_eef_abs_action_robomind_single_dataset.py with UR defaults."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output dataset root. Default follows base script behavior.",
    )
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-conversion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Actually run conversion and write files (default: false).",
    )
    parser.add_argument(
        "--recompute-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute stats for output key (default: true).",
    )
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--ee-frame-name", type=str, default="tool0")
    parser.add_argument(
        "--tcp-offset-xyz",
        type=float,
        nargs=3,
        default=DEFAULT_TCP_OFFSET_XYZ,
        help="TCP translation offset in EE frame.",
    )
    parser.add_argument(
        "--tcp-offset-rpy",
        type=float,
        nargs=3,
        default=DEFAULT_TCP_OFFSET_RPY,
        help="TCP Euler angle offset in EE frame (radians).",
    )
    parser.add_argument(
        "--tcp-offset-euler-convention",
        type=str,
        default=DEFAULT_TCP_OFFSET_EULER_CONVENTION,
        help="Euler convention used for --tcp-offset-rpy.",
    )
    parser.add_argument("--action-key", type=str, default="actions.joint_position")
    parser.add_argument("--joint-indices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--gripper-action-index", type=int, default=6)
    parser.add_argument("--output-key", type=str, default="actions.eef_abs")
    parser.add_argument(
        "--validate-fk-alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "FK alignment check is off by default because this dataset stores end-effector orientation in rotvec, "
            "while the base checker expects Euler components."
        ),
    )
    parser.add_argument("--obs-joint-key", type=str, default="observation.states.joint_position")
    parser.add_argument("--obs-eef-key", type=str, default="observation.states.end_effector")
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--precheck",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When run-conversion is enabled, run a dry-run data check first and only proceed if it passes "
            "(default: true)."
        ),
    )

    return parser.parse_known_args()


def _bool_flag(name: str, value: bool) -> str:
    return f"--{name}" if value else f"--no-{name}"


def _build_base_cmd(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        str(BASE_SCRIPT),
        "--dataset-root",
        str(args.dataset_root.resolve()),
        "--urdf-path",
        str(args.urdf_path.resolve()),
        "--ee-frame-name",
        args.ee_frame_name,
        "--tcp-offset-xyz",
        *[str(v) for v in args.tcp_offset_xyz],
        "--tcp-offset-rpy",
        *[str(v) for v in args.tcp_offset_rpy],
        "--tcp-offset-euler-convention",
        args.tcp_offset_euler_convention,
        "--action-key",
        args.action_key,
        "--joint-indices",
        *[str(i) for i in args.joint_indices],
        "--gripper-action-index",
        str(args.gripper_action_index),
        "--output-key",
        args.output_key,
        "--obs-joint-key",
        args.obs_joint_key,
        "--obs-eef-key",
        args.obs_eef_key,
        _bool_flag("run-conversion", args.run_conversion),
        _bool_flag("recompute-stats", args.recompute_stats),
        _bool_flag("validate-fk-alignment", args.validate_fk_alignment),
        _bool_flag("show-progress", args.show_progress),
    ]

    if args.output_root is not None:
        cmd.extend(["--output-root", str(args.output_root.resolve())])
    if args.overwrite_output:
        cmd.append("--overwrite-output")
    if args.dry_run:
        cmd.append("--dry-run")
    if passthrough:
        cmd.extend(passthrough)

    return cmd


def _set_bool_cli_flag(cmd: list[str], flag_name: str, value: bool) -> list[str]:
    yes = f"--{flag_name}"
    no = f"--no-{flag_name}"
    filtered = [tok for tok in cmd if tok not in {yes, no}]
    filtered.append(yes if value else no)
    return filtered


def _set_presence_flag(cmd: list[str], flag: str, enabled: bool) -> list[str]:
    filtered = [tok for tok in cmd if tok != flag]
    if enabled:
        filtered.append(flag)
    return filtered


def _run_cmd(cmd: list[str], title: str) -> None:
    print(title)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args, passthrough = parse_args()

    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(f"Base script not found: {BASE_SCRIPT}")

    cmd = _build_base_cmd(args, passthrough)

    should_precheck = args.run_conversion and not args.dry_run and args.precheck
    if should_precheck:
        precheck_cmd = _set_presence_flag(cmd, "--dry-run", True)
        precheck_cmd = _set_bool_cli_flag(precheck_cmd, "run-conversion", True)
        _run_cmd(precheck_cmd, "Precheck (dry-run) command:")

    _run_cmd(cmd, "Execution command:")


if __name__ == "__main__":
    main()
