#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT_DIR}" || exit 1
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT_DIR}"

# Keep main training processes single-threaded; let DataLoader workers use CPU.
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

DEFAULT_CONFIG_FILE="${REPO_ROOT_DIR}/latent_action_model/config/dino_base_ae.yaml"
TIMESTAMP="$(date +%m%d_%H%M%S)"
RUN_ROOT="${REPO_ROOT_DIR}/latent_action_model/logs/single_mix_runs/${TIMESTAMP}"
mkdir -p "${RUN_ROOT}"

# Default: 8 distinct single-dataset mixes (all from lam_plus_human members).
MIXES=(
  "lam_single_libero_all"
  "lam_single_bridgev2"
  "lam_single_fractal_lerobot"
  "lam_single_droid_1_0_1"
  "lam_single_agibot_merge"
  "lam_single_robomind_agilex_3rgb_delta_action"
  "lam_single_robomind_franka_1rgb_delta_action"
  "lam_single_epic_kitchens_100_lerobot"
)

# Optional override:
#   export LAM_SINGLE_MIXES="mix_a,mix_b,...,mix_h"
if [[ -n "${LAM_SINGLE_MIXES:-}" ]]; then
  IFS=',' read -r -a MIXES <<< "${LAM_SINGLE_MIXES}"
fi

if [[ "${#MIXES[@]}" -ne 8 ]]; then
  echo "LAM_SINGLE_MIXES must contain exactly 8 mix names, got ${#MIXES[@]}." >&2
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

# GPU mapping:
# - If CUDA_VISIBLE_DEVICES is set, use its ordered values.
# - Else use 0..N-1 from nvidia-smi.
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

if [[ "${#GPU_IDS[@]}" -lt 8 ]]; then
  echo "Need at least 8 visible GPUs, found ${#GPU_IDS[@]}." >&2
  exit 1
fi

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup INT TERM

echo "Starting 8 independent LAM runs"
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
      --model.task_name "lam_${mix}" \
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

echo "All 8 runs finished successfully. Logs: ${RUN_ROOT}"
