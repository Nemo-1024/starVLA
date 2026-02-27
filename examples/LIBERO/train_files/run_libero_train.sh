#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# used for check save when communication
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
export DEEPSPEED_LOG_LEVEL=error
export TRITON_CACHE_DIR=/tmp/triton_cache
mkdir -p "${TRITON_CACHE_DIR}/autotune"

# Auto-select a valid NCCL network interface if user does not provide one.
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  if ip -o link show | awk -F': ' '{print $2}' | grep -qx "bond0"; then
    export NCCL_SOCKET_IFNAME="bond0"
  else
    default_if="$(ip route | awk '/default/ {print $5; exit}')"
    if [[ -n "${default_if}" ]]; then
      export NCCL_SOCKET_IFNAME="${default_if}"
    fi
  fi
fi

# Optional IB HCA setting; keep auto when not provided.
if [[ -z "${NCCL_IB_HCA:-}" ]] && [[ -d "/sys/class/infiniband" ]]; then
  hca_list="$(ls /sys/class/infiniband 2>/dev/null | paste -sd, - || true)"
  if [[ -n "${hca_list}" ]]; then
    export NCCL_IB_HCA="${hca_list}"
  fi
fi
###########################################################################################
# Select training YAML:
# 1) first positional arg
# 2) CONFIG_YAML env
# 3) default path
config_yaml="${1:-${CONFIG_YAML:-./examples/LIBERO/train_files/starvla_latent_world_vla_libero.yaml}}"
accelerate_config="${ACCELERATE_CONFIG:-starVLA/config/accelerate/ddp_bf16.yaml}"
###########################################################################################

num_processes="${NUM_PROCESSES:-$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)}"
main_process_port="${MAIN_PROCESS_PORT:-$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)}"
if [[ ! -f "${config_yaml}" ]]; then
  echo "Config file not found: ${config_yaml}" >&2
  exit 1
fi
if [[ ! -f "${accelerate_config}" ]]; then
  echo "Accelerate config not found: ${accelerate_config}" >&2
  exit 1
fi

accelerate_cmd=(
  accelerate launch
  --config_file "${accelerate_config}"
  --num_processes "${num_processes}"
  --main_process_port "${main_process_port}"
  starVLA/training/train_starvla.py
  --config_yaml "${config_yaml}"
)

echo "Using config: ${config_yaml}"
echo "Using accelerate config: ${accelerate_config}"
echo "Using main process port: ${main_process_port}"

"${accelerate_cmd[@]}"
