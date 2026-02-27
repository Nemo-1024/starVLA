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
TIMESTAMP="$(date +%m%d_%H%M%S)"
RUN_ROOT="${REPO_ROOT_DIR}/latent_action_model/logs/diag_mix_runs/${TIMESTAMP}_node1"
mkdir -p "${RUN_ROOT}"

# Single-node leave-one-out over non-large subsets.
# BridgeV2 is excluded by default because it is already known-stable.
MIXES=(
  "lam_loo_no_big_without_libero_all"
  "lam_loo_no_big_without_fractal_lerobot"
  "lam_loo_no_big_without_robomind_agilex_3rgb_delta_action"
  "lam_loo_no_big_without_robomind_franka_1rgb_delta_action"
  "lam_loo_no_big_without_robomind_franka_3rgb_delta_action"
  "lam_loo_no_big_without_robomind_ur_1rgb"
  "lam_loo_no_big_without_robomind_franka_fr3_dual"
  "lam_loo_no_big_without_bridgev2"
)

# Optional override:
#   export LAM_LOO_MIXES="mix_a,mix_b,...,mix_h"
if [[ -n "${LAM_LOO_MIXES:-}" ]]; then
  IFS=',' read -r -a MIXES <<< "${LAM_LOO_MIXES}"
fi

echo "Selected mixes (${#MIXES[@]}): ${MIXES[*]}"

if [[ "${#MIXES[@]}" -eq 0 ]]; then
  echo "MIXES is empty." >&2
  exit 1
fi
if [[ "${#MIXES[@]}" -gt 8 ]]; then
  echo "This script is for one 8-GPU node; got ${#MIXES[@]} mixes." >&2
  exit 1
fi

HAS_USER_CONFIG=false
USER_CONFIG_FILE=""
prev_is_config=false
for arg in "$@"; do
  if [[ "${prev_is_config}" == true ]]; then
    USER_CONFIG_FILE="${arg}"
    prev_is_config=false
    continue
  fi
  if [[ "$arg" == "--config" ]] || [[ "$arg" == --config=* ]]; then
    HAS_USER_CONFIG=true
    if [[ "$arg" == "--config" ]]; then
      prev_is_config=true
      continue
    fi
    USER_CONFIG_FILE="${arg#--config=}"
  fi
done

CONFIG_ARGS=()
if [[ "${HAS_USER_CONFIG}" == false ]]; then
  CONFIG_ARGS=(--config "${DEFAULT_CONFIG_FILE}")
fi

SOURCE_CONFIG_FILE="${DEFAULT_CONFIG_FILE}"
if [[ "${HAS_USER_CONFIG}" == true && -n "${USER_CONFIG_FILE}" ]]; then
  SOURCE_CONFIG_FILE="${USER_CONFIG_FILE}"
fi
if command -v realpath >/dev/null 2>&1; then
  LAM_CONFIG_ABS="$(realpath -m "${SOURCE_CONFIG_FILE}")"
else
  LAM_CONFIG_ABS="${SOURCE_CONFIG_FILE}"
fi

GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found and CUDA_VISIBLE_DEVICES is empty." >&2
    exit 1
  fi
  GPU_COUNT="$(nvidia-smi --list-gpus | wc -l)"
  for ((i = 0; i < GPU_COUNT; i++)); do
    GPU_IDS+=("${i}")
  done
fi

if [[ "${#GPU_IDS[@]}" -lt "${#MIXES[@]}" ]]; then
  echo "Need at least ${#MIXES[@]} visible GPUs, found ${#GPU_IDS[@]}." >&2
  exit 1
fi

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup INT TERM

echo "Starting leave-one-out runs on node1"
echo "Run root: ${RUN_ROOT}"

for idx in "${!MIXES[@]}"; do
  gpu="${GPU_IDS[$idx]}"
  mix="${MIXES[$idx]}"
  run_dir="${RUN_ROOT}/gpu${gpu}_${mix}"
  log_file="${run_dir}/train.log"
  mkdir -p "${run_dir}"

  echo "[launch] gpu=${gpu} mix=${mix} log=${log_file}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export LAM_CONFIG_PATH="${LAM_CONFIG_ABS}"
    export LAM_TRAIN_LOG_FILE="${log_file}"
    python -u -m latent_action_model.main fit \
      "${CONFIG_ARGS[@]}" \
      "$@" \
      --data.data_mix "${mix}" \
      --trainer.devices 1 \
      --trainer.strategy auto \
      --trainer.logger False \
      --trainer.default_root_dir "${run_dir}" \
      --model.task_name "lam_loo_${mix}" \
      > "${log_file}" 2>&1
  ) &
  PIDS+=("$!")
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "At least one run failed. Check logs under: ${RUN_ROOT}" >&2
  exit 1
fi

echo "All node1 runs finished successfully. Logs: ${RUN_ROOT}"
