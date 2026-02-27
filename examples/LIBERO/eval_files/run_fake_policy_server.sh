#!/bin/bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export star_vla_python=${star_vla_python:-/usr/local/miniconda3/bin/python}

# Usage:
#   bash examples/LIBERO/eval_files/run_fake_policy_server.sh \
#     /mnt/project_rlinf/jlchen/datasets/libero 0 5694 8

dataset_root=${1:-/mnt/project_rlinf/jlchen/datasets/libero}
episode_index=${2:-0}
port=${3:-5694}
action_chunk_size=${4:-8}
ckpt_path=${5:-}
normalize_source=${6:-stats_gr00t}

cmd=(
  "${star_vla_python}" deployment/model_server/server_policy_dataset_replay.py
  --dataset-root "${dataset_root}"
  --episode-index "${episode_index}"
  --port "${port}"
  --action-chunk-size "${action_chunk_size}"
  --normalize-source "${normalize_source}"
  --gripper-convention neg_open
)

if [ -n "${ckpt_path}" ]; then
  cmd+=(--ckpt-path "${ckpt_path}")
fi

printf 'Running: %q ' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
