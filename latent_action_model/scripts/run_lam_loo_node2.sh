#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT_DIR}" || exit 1
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT_DIR}"

# Keep main training processes single-threaded; leave CPU for dataloader workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export TORCH_INTRAOP_THREADS=1
export TORCH_INTEROP_THREADS=1
export KMP_BLOCKTIME=0
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_DISABLE_STATS=true

HF_CACHE_BASE="$(dirname "$(dirname "${REPO_ROOT_DIR}")")/.hf_cache"
export HF_HOME="${HF_CACHE_BASE}"
export HF_DATASETS_CACHE="${HF_CACHE_BASE}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_BASE}/hub"

DEFAULT_CONFIG_FILE="${REPO_ROOT_DIR}/latent_action_model/config/dino_base_ae.yaml"
FIRST4_MIX="${LAM_FIRST4_MIX:-lam_single_droid_1_0_1}"
LAST4_MIX="${LAM_LAST4_MIX:-lam_single_agibot_merge}"
DIAG_RUNS_ROOT="${REPO_ROOT_DIR}/latent_action_model/logs/diag_mix_runs"

# Default to an independent node2 folder.
# If you explicitly want co-location with node1, set LAM_SHARED_RUN_ROOT.
RUN_ROOT="${LAM_SHARED_RUN_ROOT:-${DIAG_RUNS_ROOT}/$(date +%m%d_%H%M%S)_node2}"

# Ensure node2 outputs always live under latent_action_model/logs/diag_mix_runs.
if [[ "${RUN_ROOT}" != "${DIAG_RUNS_ROOT}"/* ]]; then
  RUN_ROOT="${DIAG_RUNS_ROOT}/$(basename "${RUN_ROOT}")"
fi
RUN_DIR_FIRST4="${RUN_ROOT}/node2_gpus0-3_${FIRST4_MIX}"
RUN_DIR_LAST4="${RUN_ROOT}/node2_gpus4-7_${LAST4_MIX}"
LOG_FILE_FIRST4="${RUN_DIR_FIRST4}/train.log"
LOG_FILE_LAST4="${RUN_DIR_LAST4}/train.log"
mkdir -p "${RUN_DIR_FIRST4}" "${RUN_DIR_LAST4}"

# Force one-node 8-GPU visibility, split into two 4-GPU jobs.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  _GPU_IDS=(0 1 2 3 4 5 6 7)
else
  IFS=',' read -r -a _GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  if [[ "${#_GPU_IDS[@]}" -lt 8 ]]; then
    echo "Need at least 8 visible GPUs for this script, got ${#_GPU_IDS[@]}." >&2
    exit 1
  fi
fi

GPU_FIRST4="${_GPU_IDS[0]},${_GPU_IDS[1]},${_GPU_IDS[2]},${_GPU_IDS[3]}"
GPU_LAST4="${_GPU_IDS[4]},${_GPU_IDS[5]},${_GPU_IDS[6]},${_GPU_IDS[7]}"
export CUDA_VISIBLE_DEVICES="${_GPU_IDS[0]},${_GPU_IDS[1]},${_GPU_IDS[2]},${_GPU_IDS[3]},${_GPU_IDS[4]},${_GPU_IDS[5]},${_GPU_IDS[6]},${_GPU_IDS[7]}"

HAS_USER_CONFIG=false
for arg in "$@"; do
  if [[ "$arg" == "--config" ]] || [[ "$arg" == --config=* ]]; then
    HAS_USER_CONFIG=true
  fi
done

CONFIG_ARGS=()
if [[ "${HAS_USER_CONFIG}" == false ]]; then
  CONFIG_ARGS=(--config "${DEFAULT_CONFIG_FILE}")
fi

for arg in "$@"; do
  if [[ "$arg" == "--data.data_mix" ]] || [[ "$arg" == --data.data_mix=* ]]; then
    echo "This script launches two different mixes; do not pass --data.data_mix via CLI." >&2
    echo "Use LAM_FIRST4_MIX and LAM_LAST4_MIX to override defaults." >&2
    exit 1
  fi
done

echo "Launching two 4-GPU trainings on one node"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Run root: ${RUN_ROOT}"
echo "First group: gpus=${GPU_FIRST4} mix=${FIRST4_MIX}"
echo "  run dir: ${RUN_DIR_FIRST4}"
echo "  log file: ${LOG_FILE_FIRST4}"
echo "Second group: gpus=${GPU_LAST4} mix=${LAST4_MIX}"
echo "  run dir: ${RUN_DIR_LAST4}"
echo "  log file: ${LOG_FILE_LAST4}"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup INT TERM

(
  export CUDA_VISIBLE_DEVICES="${GPU_FIRST4}"
  export LAM_TRAIN_LOG_FILE="${LOG_FILE_FIRST4}"
  torchrun --nproc_per_node 4 \
    --master_port "${LAM_FIRST4_MASTER_PORT:-29501}" \
    -m latent_action_model.main fit \
    "${CONFIG_ARGS[@]}" \
    "$@" \
    --data.data_mix "${FIRST4_MIX}" \
    --trainer.devices 4 \
    --trainer.logger False \
    --trainer.default_root_dir "${RUN_DIR_FIRST4}" \
    --model.task_name "lam_${FIRST4_MIX}_gpus0_3" \
    > "${LOG_FILE_FIRST4}" 2>&1
) &
PIDS+=("$!")

(
  export CUDA_VISIBLE_DEVICES="${GPU_LAST4}"
  export LAM_TRAIN_LOG_FILE="${LOG_FILE_LAST4}"
  torchrun --nproc_per_node 4 \
    --master_port "${LAM_LAST4_MASTER_PORT:-29511}" \
    -m latent_action_model.main fit \
    "${CONFIG_ARGS[@]}" \
    "$@" \
    --data.data_mix "${LAST4_MIX}" \
    --trainer.devices 4 \
    --trainer.logger False \
    --trainer.default_root_dir "${RUN_DIR_LAST4}" \
    --model.task_name "lam_${LAST4_MIX}_gpus4_7" \
    > "${LOG_FILE_LAST4}" 2>&1
) &
PIDS+=("$!")

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "At least one 4-GPU run failed. Check logs under: ${RUN_ROOT}" >&2
  exit 1
fi

echo "Both 4-GPU runs finished successfully. Logs: ${RUN_ROOT}"
