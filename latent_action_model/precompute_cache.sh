#!/bin/bash
# CPU 集群上预计算数据集统计信息的脚本

set -e  # 遇到错误立即退出

# 默认参数
DATA_ROOT_DIR="/mnt/project_rlinf/jlchen/datasets"
DATA_MIX=""
CONFIG_FILE=""
VIDEO_BACKEND="torchvision_av"
NUM_FRAMES=2
FRAME_STRIDE=1

# 帮助信息
show_help() {
    cat << EOF
用法: $0 [选项]

在 CPU 集群上预计算数据集的统计信息（stats_gr00t.json）

选项:
    -h, --help              显示此帮助信息
    -c, --config FILE       使用配置文件（YAML 格式）
    -d, --data-root DIR     数据根目录
    -m, --mix NAME          混合数据集名称
    -b, --backend BACKEND   视频后端 (torchvision_av, decord, pyav，默认: torchvision_av)
    -n, --frames NUM        帧数 (默认: 5)
    -s, --stride NUM        帧间隔 (默认: 1)

示例:
    # 使用配置文件
    $0 --config config/lam_lerobot.yaml

    # 使用命令行参数
    $0 --data-root /data/lerobot --mix bridge

    # 指定后端
    $0 --data-root /data/lerobot --mix libero --backend decord

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -d|--data-root)
            DATA_ROOT_DIR="$2"
            shift 2
            ;;
        -m|--mix)
            DATA_MIX="$2"
            shift 2
            ;;
        -b|--backend)
            VIDEO_BACKEND="$2"
            shift 2
            ;;
        -n|--frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        -s|--stride)
            FRAME_STRIDE="$2"
            shift 2
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 激活虚拟环境（如果需要）
# source /path/to/venv/bin/activate

echo "=========================================="
echo "开始预计算数据集缓存"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "主机: $(hostname)"
echo "CPU 信息: $(nproc) 核心"
echo "=========================================="

# 构建 Python 命令
PYTHON_CMD="python ${SCRIPT_DIR}/precompute_dataset_cache.py"

if [ -n "$CONFIG_FILE" ]; then
    echo "使用配置文件: $CONFIG_FILE"
    PYTHON_CMD="$PYTHON_CMD --config $CONFIG_FILE"
fi

if [ -n "$DATA_ROOT_DIR" ]; then
    echo "数据根目录: $DATA_ROOT_DIR"
    PYTHON_CMD="$PYTHON_CMD --data_root_dir $DATA_ROOT_DIR"
fi

if [ -n "$DATA_MIX" ]; then
    echo "混合数据集: $DATA_MIX"
    PYTHON_CMD="$PYTHON_CMD --data_mix $DATA_MIX"
fi

PYTHON_CMD="$PYTHON_CMD --video_backend $VIDEO_BACKEND"
PYTHON_CMD="$PYTHON_CMD --num_frames $NUM_FRAMES"
PYTHON_CMD="$PYTHON_CMD --frame_stride $FRAME_STRIDE"

echo ""
echo "执行命令:"
echo "$PYTHON_CMD"
echo "=========================================="
echo ""

# 执行
START_TIME=$(date +%s)
$PYTHON_CMD
EXIT_CODE=$?
END_TIME=$(date +%s)

ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 预计算完成!"
else
    echo "✗ 预计算失败 (退出码: $EXIT_CODE)"
fi
echo "总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

exit $EXIT_CODE
