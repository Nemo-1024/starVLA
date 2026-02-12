REPO_ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT_DIR}"

# Hugging Face 缓存目录（数据集、模型等），放在 REPO_ROOT_DIR 的祖父目录下
HF_CACHE_BASE="$(dirname "$(dirname "${REPO_ROOT_DIR}")")/.hf_cache"
export HF_HOME="${HF_CACHE_BASE}"
export HF_DATASETS_CACHE="${HF_CACHE_BASE}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_BASE}/hub"

# NOTE:
# torchrun 会启动多个训练进程；每个进程还会再启动 DataLoader workers。
# 如果这里把 OMP 线程数设得过大，很容易出现 CPU 过度订阅（表现为吞吐抖动/间歇性卡顿、进度条跳步）。
# 因此默认将各类 BLAS/OMP 线程限制为 1；如需手动调优，可在外部先 export 覆盖。
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
export TORCH_NCCL_TIMEOUT=1800   # 单位：秒




# 训练参数
# CONFIG_FILE="config/lam-vjepa_large.yaml"
# 默认配置文件（当未在命令行通过 --config 指定时使用）
DEFAULT_CONFIG_FILE="${REPO_ROOT_DIR}/latent_action_model/config/dino_base_ae.yaml"
TIMESTAMP="$(date +%m%d_%H%M%S)"
LOG_DIR="latent_action_model/logs/train_logs/${TIMESTAMP}"

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

# 仅在用户通过 --config 指定配置时，备份该用户配置到日志目录
if [[ "${HAS_USER_CONFIG}" == true ]]; then
    if [[ -n "${USER_CONFIG_FILE}" && -f "${USER_CONFIG_FILE}" ]]; then
        cp "${USER_CONFIG_FILE}" "${LOG_DIR}/$(basename "${USER_CONFIG_FILE}")"
    else
        echo "⚠️ 用户配置文件未找到或未提供: ${USER_CONFIG_FILE}" >&2
    fi
fi

# 若指定了用户配置，将其绝对路径导出为环境变量，供回调保存使用
if [[ "${HAS_USER_CONFIG}" == true && -n "${USER_CONFIG_FILE}" ]]; then
    if [[ "${USER_CONFIG_FILE}" = /* ]]; then
        export LAM_CONFIG_PATH="${USER_CONFIG_FILE}"
    else
        if command -v realpath &> /dev/null; then
            export LAM_CONFIG_PATH="$(realpath -m "${USER_CONFIG_FILE}")"
        else
            export LAM_CONFIG_PATH="${REPO_ROOT_DIR}/${USER_CONFIG_FILE}"
        fi
    fi
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
