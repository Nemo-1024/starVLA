#!/usr/bin/env bash
# 测试不同 num_workers 和 prefetch_factor 下，训练 64 次迭代的耗时
# 用法: 在 latent_action_model 的上级目录执行，或设置 REPO_ROOT_DIR
#   cd /mnt/project_rlinf/jlchen/code/starVLA && bash latent_action_model/scripts/benchmark_dataloader_64iters.sh

set -e

REPO_ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT_DIR}"
cd "${REPO_ROOT_DIR}" || exit 1

# 与 train.sh 一致的环境变量，避免 CPU 过载
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
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TIMEOUT=720000

HF_CACHE_BASE="$(dirname "$(dirname "${REPO_ROOT_DIR}")")/.hf_cache"
export HF_HOME="${HF_CACHE_BASE}"
export HF_DATASETS_CACHE="${HF_CACHE_BASE}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_BASE}/hub"

CONFIG="${REPO_ROOT_DIR}/latent_action_model/config/dino_base_ae.yaml"
LIMIT_BATCHES=64

# 使用所有可见 GPU（与 train.sh 一致）；若只测单卡可先 export CUDA_VISIBLE_DEVICES=0
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
fi
[ "${NUM_GPUS}" -lt 1 ] && NUM_GPUS=1

# 要测试的 (num_workers, prefetch_factor) 组合
# 格式: "num_workers prefetch_factor"
COMBOS=(
    "2  4"
    "4  4"
    "4  8"
    "8  4"
    "8  8"
    "8  16"
    "12 8"
    "16 8"
)

RESULTS_FILE="${REPO_ROOT_DIR}/latent_action_model/logs/benchmark_64iters_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p "$(dirname "${RESULTS_FILE}")"

echo "========================================"
echo "Benchmark: 64 training iterations"
echo "Config: ${CONFIG}"
echo "limit_train_batches: ${LIMIT_BATCHES}"
echo "GPUs: ${NUM_GPUS}"
echo "Results: ${RESULTS_FILE}"
echo "========================================"

{
    echo "num_workers,prefetch_factor,time_sec,time_hms"
} >> "${RESULTS_FILE}"

for combo in "${COMBOS[@]}"; do
    read -r nw pf <<< "${combo}"
    echo "----------------------------------------"
    echo "Running num_workers=${nw} prefetch_factor=${pf} ..."
    echo "(首次加载数据/模型可能较慢，64 步约需数分钟，请观察进度条)"
    t0=$SECONDS
    # 直接输出到终端，便于看到进度条、确认在运行（不再用 tail 避免“卡住”错觉）
    torchrun --nproc_per_node "${NUM_GPUS}" \
        -m latent_action_model.main fit \
        --config "${CONFIG}" \
        --trainer.limit_train_batches "${LIMIT_BATCHES}" \
        --trainer.devices "${NUM_GPUS}" \
        --trainer.log_every_n_steps 16 \
        --trainer.num_sanity_val_steps 0 \
        --data.num_workers "${nw}" \
        --data.prefetch_factor "${pf}"
    elapsed=$((SECONDS - t0))
    # 格式化 H:MM:SS
    h=$((elapsed / 3600))
    m=$(((elapsed % 3600) / 60))
    s=$((elapsed % 60))
    if [ "$h" -gt 0 ]; then
        hms=$(printf "%d:%02d:%02d" "$h" "$m" "$s")
    else
        hms=$(printf "%d:%02d" "$m" "$s")
    fi
    echo "  -> ${elapsed}s (${hms})"
    echo "${nw},${pf},${elapsed},${hms}" >> "${RESULTS_FILE}"
done

echo "========================================"
echo "Results summary (saved to ${RESULTS_FILE}):"
column -t -s',' "${RESULTS_FILE}"
echo "========================================"
