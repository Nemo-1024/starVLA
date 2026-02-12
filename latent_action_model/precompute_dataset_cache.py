#!/usr/bin/env python3
"""
独立脚本：在 CPU 集群上预计算数据集的统计信息

此脚本会触发数据集的初始化，生成以下缓存文件：
1. meta/stats_gr00t.json - 数据集统计信息

使用方法：
    python precompute_dataset_cache.py --data_root_dir /path/to/data --data_mix bridge
    
或使用配置文件：
    python precompute_dataset_cache.py --config config/lam_lerobot.yaml
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG


def build_modality_config(
    robot_type: str,
    num_frames: int = 5,
    frame_stride: int = 1,
    preferred_video_key: Optional[str] = None,
    state_keys: Optional[list[str]] = None,
) -> dict[str, ModalityConfig]:
    """构建 modality 配置"""
    base_cfg = ROBOT_TYPE_CONFIG_MAP[robot_type].modality_config()
    
    delta_indices = list(range(0, num_frames * frame_stride, frame_stride))
    
    # video
    video_keys = base_cfg["video"].modality_keys
    if preferred_video_key and preferred_video_key in video_keys:
        video_keys = [preferred_video_key]
    video_modality = ModalityConfig(delta_indices=delta_indices, modality_keys=video_keys)
    
    # state
    state_modality_keys = list(state_keys) if state_keys else base_cfg["state"].modality_keys
    state_modality = ModalityConfig(delta_indices=delta_indices, modality_keys=state_modality_keys)
    
    return {
        "video": video_modality,
        "state": state_modality,
    }


def precompute_single_dataset(
    dataset_path: Path,
    robot_type: str,
    video_backend: str = "torchvision_av",
    num_frames: int = 5,
    frame_stride: int = 1,
    preferred_video_key: Optional[str] = None,
    state_keys: Optional[list[str]] = None,
) -> bool:
    """
    预计算单个数据集的缓存
    
    返回：
        bool: 是否成功
    """
    print(f"\n{'='*80}")
    print(f"预处理数据集: {dataset_path.name}")
    print(f"机器人类型: {robot_type}")
    print(f"{'='*80}\n")
    
    if not dataset_path.exists():
        print(f"警告: 数据集路径不存在: {dataset_path}")
        return False
    
    try:
        # 构建 modality 配置
        modality_cfg = build_modality_config(
            robot_type=robot_type,
            num_frames=num_frames,
            frame_stride=frame_stride,
            preferred_video_key=preferred_video_key,
            state_keys=state_keys,
        )
        
        # 获取 embodiment tag
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG.get(robot_type)
        if embodiment_tag is None:
            raise ValueError(f"未找到机器人类型 '{robot_type}' 的 embodiment tag")
        
        # 创建数据集配置
        data_cfg = {
            "data_root_dir": str(dataset_path.parent),
            "video_backend": video_backend,
        }
        
        print(f"开始初始化数据集...")
        start_time = time.time()
        
        # 初始化数据集 - 这会触发统计信息计算
        dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=modality_cfg,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            data_cfg=data_cfg,
        )
        
        elapsed_time = time.time() - start_time
        
        # 验证缓存文件是否生成
        stats_path = dataset_path / "meta" / "stats_gr00t.json"
        
        print(f"\n{'='*80}")
        print(f"数据集初始化完成!")
        print(f"耗时: {elapsed_time:.2f} 秒")
        print(f"数据集长度: {len(dataset)}")
        print(f"轨迹数量: {len(dataset.trajectory_ids)}")
        print(f"\n生成的缓存文件:")
        print(f"  - 统计信息: {stats_path} ({'存在' if stats_path.exists() else '不存在'})")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"错误: 处理数据集 {dataset_path.name} 时出错:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def precompute_mixture(
    data_root_dir: Path,
    data_mix: str,
    video_backend: str = "torchvision_av",
    num_frames: int = 5,
    frame_stride: int = 1,
    preferred_video_key: Optional[str] = None,
    state_keys: Optional[list[str]] = None,
):
    """预计算混合数据集中所有数据集的缓存"""
    
    print(f"\n{'#'*80}")
    print(f"# 开始预处理混合数据集: {data_mix}")
    print(f"# 数据根目录: {data_root_dir}")
    print(f"{'#'*80}\n")
    
    if data_mix not in DATASET_NAMED_MIXTURES:
        print(f"错误: 未知的混合数据集名称 '{data_mix}'")
        print(f"可用的混合数据集: {list(DATASET_NAMED_MIXTURES.keys())}")
        return
    
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    
    # 去重 (dataset_name, robot_type)
    seen = set()
    datasets_to_process = []
    for dataset_name, weight, robot_type in mixture_spec:
        key = (dataset_name, robot_type)
        if key in seen:
            continue
        seen.add(key)
        datasets_to_process.append((dataset_name, robot_type, weight))
    
    print(f"混合数据集包含 {len(datasets_to_process)} 个唯一数据集:\n")
    for dataset_name, robot_type, weight in datasets_to_process:
        print(f"  - {dataset_name:50s} (robot_type={robot_type:20s}, weight={weight:.4f})")
    print()
    
    # 处理每个数据集
    success_count = 0
    failed_datasets = []
    
    for idx, (dataset_name, robot_type, weight) in enumerate(datasets_to_process, 1):
        print(f"\n进度: [{idx}/{len(datasets_to_process)}]")
        dataset_path = data_root_dir / dataset_name
        
        success = precompute_single_dataset(
            dataset_path=dataset_path,
            robot_type=robot_type,
            video_backend=video_backend,
            num_frames=num_frames,
            frame_stride=frame_stride,
            preferred_video_key=preferred_video_key,
            state_keys=state_keys,
        )
        
        if success:
            success_count += 1
        else:
            failed_datasets.append(dataset_name)
    
    # 打印总结
    print(f"\n{'#'*80}")
    print(f"# 预处理完成!")
    print(f"# 成功: {success_count}/{len(datasets_to_process)}")
    if failed_datasets:
        print(f"# 失败的数据集:")
        for name in failed_datasets:
            print(f"#   - {name}")
    print(f"{'#'*80}\n")


def load_config_from_yaml(config_path: Path) -> dict:
    """从 YAML 配置文件加载参数"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 提取 data 部分的参数
    if 'data' in config:
        data_config = config['data']
        # 处理嵌套的 init_args
        if 'init_args' in data_config:
            return data_config['init_args']
        return data_config
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="预计算数据集的统计信息和索引缓存",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用命令行参数
    python precompute_dataset_cache.py --data_root_dir /data/lerobot --data_mix bridge
    
    # 使用配置文件
    python precompute_dataset_cache.py --config config/lam_lerobot.yaml
    
    # 处理单个数据集
    python precompute_dataset_cache.py --data_root_dir /data/lerobot --dataset_name bridge_dataset --robot_type widowx
        """
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML 配置文件路径（与训练配置相同的格式）"
    )
    parser.add_argument(
        "--data_root_dir",
        type=Path,
        help="数据根目录路径"
    )
    parser.add_argument(
        "--data_mix",
        type=str,
        help="混合数据集名称 (例如: bridge, libero, oxe 等)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="单个数据集名称（用于只处理一个数据集）"
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        help="机器人类型（配合 --dataset_name 使用）"
    )
    parser.add_argument(
        "--video_backend",
        type=str,
        default="torchvision_av",
        choices=["torchvision_av", "decord", "pyav"],
        help="视频读取后端 (默认: torchvision_av)"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=5,
        help="帧数 (默认: 5)"
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="帧间隔 (默认: 1)"
    )
    parser.add_argument(
        "--preferred_video_key",
        type=str,
        help="首选的视频键名"
    )
    parser.add_argument(
        "--state_keys",
        type=str,
        nargs="+",
        help="状态键列表"
    )
    args = parser.parse_args()
    
    # 如果提供了配置文件，从中加载参数
    if args.config:
        print(f"从配置文件加载参数: {args.config}")
        config = load_config_from_yaml(args.config)
        
        # 用配置文件的值填充缺失的参数
        if not args.data_root_dir and 'data_root_dir' in config:
            args.data_root_dir = Path(config['data_root_dir'])
        if not args.data_mix and 'data_mix' in config:
            args.data_mix = config['data_mix']
        if 'video_backend' in config:
            args.video_backend = config.get('video_backend', args.video_backend)
        if 'num_frames' in config:
            args.num_frames = config.get('num_frames', args.num_frames)
        if 'frame_stride' in config:
            args.frame_stride = config.get('frame_stride', args.frame_stride)
        if 'preferred_video_key' in config:
            args.preferred_video_key = config.get('preferred_video_key', args.preferred_video_key)
        if 'state_keys' in config:
            args.state_keys = config.get('state_keys', args.state_keys)
    
    # 验证必需参数
    if not args.data_root_dir:
        parser.error("必须提供 --data_root_dir 或 --config")
    
    if not args.data_mix and not args.dataset_name:
        parser.error("必须提供 --data_mix 或 --dataset_name")
    
    # 处理单个数据集
    if args.dataset_name:
        if not args.robot_type:
            parser.error("处理单个数据集时必须提供 --robot_type")
        
        dataset_path = args.data_root_dir / args.dataset_name
        precompute_single_dataset(
            dataset_path=dataset_path,
            robot_type=args.robot_type,
            video_backend=args.video_backend,
            num_frames=args.num_frames,
            frame_stride=args.frame_stride,
            preferred_video_key=args.preferred_video_key,
            state_keys=args.state_keys,
        )
    else:
        # 处理混合数据集
        precompute_mixture(
            data_root_dir=args.data_root_dir,
            data_mix=args.data_mix,
            video_backend=args.video_backend,
            num_frames=args.num_frames,
            frame_stride=args.frame_stride,
            preferred_video_key=args.preferred_video_key,
            state_keys=args.state_keys,
        )


if __name__ == "__main__":
    main()
