REPO_ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT_DIR}"
cd "${REPO_ROOT_DIR}" || exit 1

# Hugging Face 缓存目录（数据集、模型等），放在 REPO_ROOT_DIR 的祖父目录下
HF_CACHE_BASE="$(dirname "$(dirname "${REPO_ROOT_DIR}")")/.hf_cache"
export HF_HOME="${HF_CACHE_BASE}"
export HF_DATASETS_CACHE="${HF_CACHE_BASE}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_BASE}/hub"

# NOTE:
# 主进程（各 GPU 训练进程）限制为单线程，把 CPU 留给 DataLoader workers，避免过度订阅导致卡顿。
# DataLoader worker 内线程数由 lerobot_datamodule 的 worker_init_fn 控制（默认 1；可设 LAM_WORKER_OMP_THREADS=2 进一步压榨）。
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export TORCH_INTRAOP_THREADS=1
export TORCH_INTEROP_THREADS=1
# OpenMP: 不等待，尽快让出 CPU 给其他线程/进程（利于多 worker 数据加载）
export KMP_BLOCKTIME=0
# 可选：为 worker 内解码留更多线程（默认不设=1）；若 CPU 仍有大量 idle 可试 export LAM_WORKER_OMP_THREADS=2
# export LAM_WORKER_OMP_THREADS=2
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_DISABLE_STATS=true
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TIMEOUT=720000  # 单位：秒




# 训练参数
# CONFIG_FILE="config/lam-vjepa_large.yaml"
# 默认配置文件（当未在命令行通过 --config 指定时使用）
DEFAULT_CONFIG_FILE="${REPO_ROOT_DIR}/latent_action_model/config/dino_base_ae.yaml"
TIMESTAMP="$(date +%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT_DIR}/latent_action_model/logs/train_logs/${TIMESTAMP}"

LOG_FILE="${LOG_DIR}/train_logs.log"

# 可选：从检查点恢复（两种方式）
CKPT_PATH="${CKPT_PATH:-}"

# W&B 配置
export WANDB_API_KEY="8d44fb58134f3f96e048d943a2543c51ff4f1d09"
export WANDB_DIR="${REPO_ROOT_DIR}/latent_action_model"

# export WANDB_MODE="offline"

# 选择配置：若命令行包含 --config，则使用用户指定配置；否则使用默认配置
HAS_USER_CONFIG=false
for arg in "$@"; do
    if [[ "$arg" == "--config" ]] || [[ "$arg" == --config=* ]]; then
        HAS_USER_CONFIG=true
        break
    fi
done

# 若用户提供 --config，则解析出其指定的配置文件路径
USER_CONFIG_FILE=""
if [[ "$HAS_USER_CONFIG" == true ]]; then
    prev_is_config=false
    for arg in "$@"; do
        if [[ "$prev_is_config" == true ]]; then
            USER_CONFIG_FILE="$arg"
            prev_is_config=false
            continue
        fi
        if [[ "$arg" == "--config" ]]; then
            prev_is_config=true
            continue
        fi
        if [[ "$arg" == --config=* ]]; then
            USER_CONFIG_FILE="${arg#--config=}"
        fi
    done
fi

if [[ "$HAS_USER_CONFIG" == true ]]; then
    CONFIG_CLI=""
    CONFIG_SHOWN="${USER_CONFIG_FILE:-命令行(--config)已指定}"
else
    CONFIG_CLI="--config ${DEFAULT_CONFIG_FILE}"
    CONFIG_SHOWN="${DEFAULT_CONFIG_FILE}"
fi

echo "🚀 美好的事情发生了！！！"
echo "🚀 开始 VJEPA_LAM 训练..."
echo "📋 配置文件: ${CONFIG_SHOWN}"
echo "📝 日志文件: ${LOG_FILE}"
echo "🗂️ W&B 目录: ${WANDB_DIR}/wandb"
if [[ -n "${CKPT_PATH}" ]]; then
    echo "🔁 从检查点恢复: ${CKPT_PATH}"
fi

# 确保日志目录存在
mkdir -p "${LOG_DIR}"
mkdir -p "${WANDB_DIR}/wandb"

# 导出外部日志文件路径，供 Lightning 回调在训练结束后复制到 checkpoints 父目录
if command -v realpath &> /dev/null; then
    export LAM_TRAIN_LOG_FILE="$(realpath -m "${LOG_FILE}")"
else
    export LAM_TRAIN_LOG_FILE="${REPO_ROOT_DIR}/${LOG_FILE}"
fi

# 解析本次训练实际使用的配置文件路径（无论默认还是用户 --config）
SOURCE_CONFIG_FILE="${DEFAULT_CONFIG_FILE}"
if [[ "${HAS_USER_CONFIG}" == true && -n "${USER_CONFIG_FILE}" ]]; then
    SOURCE_CONFIG_FILE="${USER_CONFIG_FILE}"
fi

if [[ "${SOURCE_CONFIG_FILE}" = /* ]]; then
    export LAM_CONFIG_PATH="${SOURCE_CONFIG_FILE}"
else
    if command -v realpath &> /dev/null; then
        export LAM_CONFIG_PATH="$(realpath -m "${SOURCE_CONFIG_FILE}")"
    else
        export LAM_CONFIG_PATH="${REPO_ROOT_DIR}/${SOURCE_CONFIG_FILE}"
    fi
fi

# 备份实际配置到外部训练日志目录，便于从 train_logs 目录直接回溯参数
if [[ -f "${LAM_CONFIG_PATH}" ]]; then
    cp "${LAM_CONFIG_PATH}" "${LOG_DIR}/$(basename "${LAM_CONFIG_PATH}")"
else
    echo "⚠️ 配置文件未找到: ${LAM_CONFIG_PATH}" >&2
fi

# 自动获取 GPU 数量
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    # fallback: 使用 torch 获取 GPU 数量
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
fi
echo "🖥️ 检测到 GPU 数量: ${NUM_GPUS}"

# --- 启动训练 ---
# torchrun 负责启动进程和设置通信
# LightningCLI (--trainer.*) 负责配置 Trainer 对象
# 使用 unbuffered 输出确保日志实时写入
torchrun --nproc_per_node ${NUM_GPUS} \
         -m latent_action_model.main fit \
         ${CONFIG_CLI} \
         ${CKPT_PATH:+--ckpt_path ${CKPT_PATH}} \
         "$@" \
         2>&1 | tee -a ${LOG_FILE}

# 训练结束后确保日志文件存在
if [ -f "${LOG_FILE}" ]; then
    echo "✅ 训练日志已保存到: ${LOG_FILE}"
    echo "📊 日志文件大小: $(du -h ${LOG_FILE} | cut -f1)"
else
    echo "⚠️ 警告: 日志文件未找到: ${LOG_FILE}"
fi
