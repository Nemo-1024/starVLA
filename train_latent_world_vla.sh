#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Keep runtime caches under dataset root by default (avoid system paths).
DEFAULT_DATASET_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)/datasets"
DATASET_ROOT_DIR="${DATASET_ROOT_DIR:-${DEFAULT_DATASET_ROOT}}"

# ---------------------- Default configuration ----------------------
DEFAULT_CONFIG_YAML="starVLA/config/training/starvla_train_latent_world_vla_independent.yaml"
DEFAULT_ACCELERATE_CONFIG="starVLA/config/accelerate/ddp_bf16.yaml"

CONFIG_YAML="${CONFIG_YAML:-${DEFAULT_CONFIG_YAML}}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${DEFAULT_ACCELERATE_CONFIG}}"

# Use first positional argument as config path if provided.
if [[ $# -gt 0 && "${1}" != --* ]]; then
  CONFIG_YAML="$1"
  shift
fi

# ---------------------- Runtime environment -----------------------
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
# Prefer the non-deprecated PyTorch NCCL env var, but accept legacy input.
if [[ -n "${NCCL_ASYNC_ERROR_HANDLING:-}" && -z "${TORCH_NCCL_ASYNC_ERROR_HANDLING:-}" ]]; then
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING}"
else
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
fi
unset NCCL_ASYNC_ERROR_HANDLING
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"
export DEEPSPEED_LOG_LEVEL="${DEEPSPEED_LOG_LEVEL:-error}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"

# Accept legacy TRANSFORMERS_CACHE input once, then switch to HF_HOME layout.
if [[ -n "${TRANSFORMERS_CACHE:-}" && -z "${HF_HOME:-}" ]]; then
  export HF_HOME="$(dirname "${TRANSFORMERS_CACHE}")"
fi
export HF_HOME="${HF_HOME:-${DATASET_ROOT_DIR}/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
unset TRANSFORMERS_CACHE
mkdir -p "${TRITON_CACHE_DIR}/autotune"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}"

# Auto-select NCCL network interface if user does not provide one.
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  if ip -o link show | awk -F': ' '{print $2}' | grep -qx "bond0" >/dev/null 2>&1; then
    export NCCL_SOCKET_IFNAME="bond0"
  else
    default_if="$(ip route | awk '/default/ {print $5; exit}')"
    if [[ -n "${default_if}" ]]; then
      export NCCL_SOCKET_IFNAME="${default_if}"
    fi
  fi
fi

# Optional IB HCA setting; keep auto when not provided.
if [[ -z "${NCCL_IB_HCA:-}" && -d "/sys/class/infiniband" ]]; then
  hca_list="$(ls /sys/class/infiniband 2>/dev/null | paste -sd, - || true)"
  if [[ -n "${hca_list}" ]]; then
    export NCCL_IB_HCA="${hca_list}"
  fi
fi

# ---------------------- Argument and file checks ------------------
if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "Config file not found: ${CONFIG_YAML}" >&2
  exit 1
fi

if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "Accelerate config not found: ${ACCELERATE_CONFIG}" >&2
  exit 1
fi

NUM_PROCESSES="${NUM_PROCESSES:-$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)}"

MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)}"

# ---------------------- Logging -----------------------------------
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/results/train_logs/latent_world_vla/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train.log"

echo "Starting LatentWorldVLA training"
echo "Repo root: ${REPO_ROOT}"
echo "Config YAML: ${CONFIG_YAML}"
echo "Accelerate config: ${ACCELERATE_CONFIG}"
echo "Num processes: ${NUM_PROCESSES}"
echo "Main process port: ${MAIN_PROCESS_PORT}"
echo "TORCH_NCCL_ASYNC_ERROR_HANDLING: ${TORCH_NCCL_ASYNC_ERROR_HANDLING}"
echo "HF cache root: ${HF_HOME}"
echo "HF hub cache: ${HF_HUB_CACHE}"
echo "HF datasets cache: ${HF_DATASETS_CACHE}"
echo "Log file: ${LOG_FILE}"

cp "${CONFIG_YAML}" "${LOG_DIR}/$(basename "${CONFIG_YAML}")"

accelerate_cmd=(
  accelerate launch
  --config_file "${ACCELERATE_CONFIG}"
  --num_processes "${NUM_PROCESSES}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  starVLA/training/train_starvla.py
  --config_yaml "${CONFIG_YAML}"
)

# Forward any extra CLI overrides, e.g.:
#   bash train_latent_world_vla.sh --trainer.max_train_steps 1000
if [[ $# -gt 0 ]]; then
  accelerate_cmd+=("$@")
fi

"${accelerate_cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
