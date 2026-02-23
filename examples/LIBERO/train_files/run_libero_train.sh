#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000

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
# === Please modify the following paths according to your environment ===
Framework_name="${FRAMEWORK_NAME:-LatentWorldVLAIndependent}"
freeze_module_list="${FREEZE_MODULE_LIST:-}"
base_vlm="${BASE_VLM:-/mnt/project_rlinf/jlchen/weights/Qwen3-VL-2B-Instruct}"
config_yaml="${CONFIG_YAML:-./examples/LIBERO/train_files/starvla_latent_world_vla_libero.yaml}"
libero_data_root="${LIBERO_DATA_ROOT:-/mnt/project_rlinf/jlchen/datasets}"
data_mix="${DATA_MIX:-libero}"
video_backend="${VIDEO_BACKEND:-pyav}"
run_root_dir="${RUN_ROOT_DIR:-./results/Checkpoints}"
run_id="${RUN_ID:-1229_libero_latent_world_vla}"
# === End of environment variable configuration ===
###########################################################################################

num_processes="${NUM_PROCESSES:-$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)}"
per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-16}"
max_train_steps="${MAX_TRAIN_STEPS:-80000}"
save_interval="${SAVE_INTERVAL:-10000}"
logging_frequency="${LOGGING_FREQUENCY:-100}"
eval_interval="${EVAL_INTERVAL:-100}"

# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp "${SCRIPT_PATH}" "${output_dir}/"


accelerate_cmd=(
  accelerate launch
  --config_file starVLA/config/accelerate/ddp_bf16.yaml
  --num_processes "${num_processes}"
  starVLA/training/train_starvla.py
  --config_yaml "${config_yaml}"
  --framework.name "${Framework_name}"
  --framework.qwenvl.base_vlm "${base_vlm}"
  --datasets.vla_data.data_root_dir "${libero_data_root}"
  --datasets.vla_data.data_mix "${data_mix}"
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}"
  --datasets.vla_data.video_backend "${video_backend}"
  --trainer.max_train_steps "${max_train_steps}"
  --trainer.save_interval "${save_interval}"
  --trainer.logging_frequency "${logging_frequency}"
  --trainer.eval_interval "${eval_interval}"
  --run_root_dir "${run_root_dir}"
  --run_id "${run_id}"
  --wandb_project starVLA_Libero
)

if [[ -n "${freeze_module_list}" ]]; then
  accelerate_cmd+=(--trainer.freeze_modules "${freeze_module_list}")
fi

"${accelerate_cmd[@]}"



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/accelerate/ddp_bf16.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
