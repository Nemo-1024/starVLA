#!/usr/bin/env python3
"""
统计数据集中缺失语言标注的轨迹数目（高性能版）

此脚本会分析数据集中有多少轨迹缺失语言标注，帮助判断：
1. 缺失语言轨迹的比例
2. 是否有必要过滤这些轨迹
3. 哪些数据集存在语言缺失问题

优化：直接从 parquet 文件读取，避免加载整个 HF Dataset

使用方法：
    python analyze_missing_language.py --data_root_dir /path/to/data --data_mix lam --num_workers 32
    python analyze_missing_language.py --config config/test.yaml --num_workers 16
"""

import argparse
import sys
import json
import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import multiprocessing as mp
import glob

import yaml
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _analyze_dataset_fast(args: Tuple) -> Dict:
    """
    快速分析单个数据集的语言缺失情况
    
    直接从 parquet 文件读取，完全绑过 HF Dataset 加载
    """
    dataset_path_str, robot_type, verbose = args
    dataset_path = Path(dataset_path_str)
    
    result = {
        "dataset_name": dataset_path.name,
        "robot_type": robot_type,
        "total_trajectories": 0,
        "missing_language_trajectories": 0,
        "read_error_trajectories": 0,
        "valid_trajectories": 0,
        "missing_language_ratio": 0.0,
        "missing_trajectory_ids": [],
        "error_trajectory_ids": [],
        "sample_missing_languages": [],
        "status": "success",
        "error_message": None,
    }
    
    if not dataset_path.exists():
        result["status"] = "error"
        result["error_message"] = f"数据集路径不存在: {dataset_path}"
        return result
    
    try:
        # ========== 方法 1: 直接从 parquet 文件读取（最快） ==========
        
        # 1. 读取 tasks.parquet
        tasks_path = dataset_path / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            result["status"] = "error"
            result["error_message"] = f"tasks.parquet 不存在: {tasks_path}"
            return result
        
        tasks_df = pd.read_parquet(tasks_path)
        # 检查 tasks 表中哪些 task 语言为空
        if "task" not in tasks_df.columns:
            result["status"] = "error"
            result["error_message"] = "tasks.parquet 缺少 'task' 列"
            return result
        
        # task_index -> task (语言)
        tasks_df = tasks_df.reset_index()
        if "task_index" not in tasks_df.columns and "index" in tasks_df.columns:
            tasks_df = tasks_df.rename(columns={"index": "task_index"})
        
        # 找出空语言的 task_index
        empty_task_indices = set()
        for _, row in tasks_df.iterrows():
            task_idx = row.get("task_index", row.name)
            task_text = row.get("task", "")
            if not task_text or task_text == "":
                empty_task_indices.add(task_idx)
        
        # 2. 读取 episodes parquet 文件
        episodes_pattern = dataset_path / "meta" / "episodes" / "*" / "*.parquet"
        episodes_files = glob.glob(str(episodes_pattern))
        
        if not episodes_files:
            # 尝试 v2.0 格式
            episodes_jsonl = dataset_path / "meta" / "episodes.jsonl"
            if episodes_jsonl.exists():
                result["status"] = "error"
                result["error_message"] = "数据集是 v2.0 格式，请先转换为 v3.0"
                return result
            result["status"] = "error"
            result["error_message"] = f"找不到 episodes parquet 文件: {episodes_pattern}"
            return result
        
        # 读取所有 episodes parquet 并合并
        episodes_dfs = []
        for ep_file in episodes_files:
            episodes_dfs.append(pd.read_parquet(ep_file))
        episodes_df = pd.concat(episodes_dfs, ignore_index=True)
        
        result["total_trajectories"] = len(episodes_df)
        
        # 3. 检查每个 episode 的 task_index
        # 需要从 data parquet 文件获取 task_index（因为 episodes 可能没有）
        
        # 首先检查 episodes_df 是否有 task_index 列
        if "task_index" in episodes_df.columns:
            # 直接使用 episodes 的 task_index
            for _, row in episodes_df.iterrows():
                ep_idx = row["episode_index"]
                task_idx = row["task_index"]
                
                if task_idx in empty_task_indices:
                    result["missing_language_trajectories"] += 1
                    result["missing_trajectory_ids"].append(int(ep_idx))
                    if len(result["sample_missing_languages"]) < 10:
                        result["sample_missing_languages"].append({
                            "trajectory_id": int(ep_idx),
                            "task_index": int(task_idx) if pd.notna(task_idx) else "N/A",
                        })
                else:
                    result["valid_trajectories"] += 1
        else:
            # 需要从 data parquet 获取 task_index
            # 只读取第一行每个 episode 的 task_index（节省内存）
            
            data_pattern = dataset_path / "data" / "*" / "*.parquet"
            data_files = sorted(glob.glob(str(data_pattern)))
            
            if not data_files:
                result["status"] = "error"
                result["error_message"] = f"找不到 data parquet 文件: {data_pattern}"
                return result
            
            # 使用 pyarrow 高效读取特定列
            episode_task_map = {}
            
            for data_file in tqdm(data_files, desc=f"扫描 {dataset_path.name}", leave=False, disable=not verbose):
                # 只读取 episode_index 和 task_index 两列
                table = pq.read_table(data_file, columns=["episode_index", "task_index"])
                df = table.to_pandas()
                
                # 每个 episode 取第一个 task_index
                for ep_idx, group in df.groupby("episode_index"):
                    if ep_idx not in episode_task_map:
                        episode_task_map[ep_idx] = group["task_index"].iloc[0]
            
            # 检查每个 episode
            for ep_idx, task_idx in episode_task_map.items():
                if pd.isna(task_idx) or task_idx in empty_task_indices:
                    result["missing_language_trajectories"] += 1
                    result["missing_trajectory_ids"].append(int(ep_idx))
                    if len(result["sample_missing_languages"]) < 10:
                        result["sample_missing_languages"].append({
                            "trajectory_id": int(ep_idx),
                            "task_index": int(task_idx) if pd.notna(task_idx) else "N/A",
                        })
                else:
                    # 还需要检查 task 文本是否为空
                    task_text = tasks_df[tasks_df["task_index"] == task_idx]["task"].values
                    if len(task_text) == 0 or not task_text[0] or task_text[0] == "":
                        result["missing_language_trajectories"] += 1
                        result["missing_trajectory_ids"].append(int(ep_idx))
                        if len(result["sample_missing_languages"]) < 10:
                            result["sample_missing_languages"].append({
                                "trajectory_id": int(ep_idx),
                                "task_index": int(task_idx),
                            })
                    else:
                        result["valid_trajectories"] += 1
        
        # 计算比例
        if result["total_trajectories"] > 0:
            result["missing_language_ratio"] = result["missing_language_trajectories"] / result["total_trajectories"]
        
        # 限制返回的 ID 列表长度
        result["missing_trajectory_ids"] = result["missing_trajectory_ids"][:100]
        result["error_trajectory_ids"] = result["error_trajectory_ids"][:100]
        
    except Exception as e:
        result["status"] = "error"
        result["error_message"] = str(e)
        import traceback
        result["error_message"] += "\n" + traceback.format_exc()
    
    return result


def analyze_single_dataset(
    dataset_path: Path,
    robot_type: str,
    video_backend: str = "torchvision_av",
    verbose: bool = False,
    num_workers: int = 1,
) -> dict:
    """
    分析单个数据集的语言缺失情况
    """
    return _analyze_dataset_fast((str(dataset_path), robot_type, verbose))


def analyze_mixture(
    data_root_dir: Path,
    data_mix: str,
    video_backend: str = "torchvision_av",
    verbose: bool = False,
    output_file: Optional[str] = None,
    num_workers: int = 1,
):
    """分析混合数据集中所有数据集的语言缺失情况（多进程并行）"""
    
    from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
    
    print(f"\n{'#'*80}")
    print(f"# 分析混合数据集: {data_mix}")
    print(f"# 数据根目录: {data_root_dir}")
    print(f"# 并行进程数: {num_workers}")
    print(f"# 模式: 快速 parquet 直读（跳过 HF Dataset 加载）")
    print(f"{'#'*80}\n")
    
    if data_mix not in DATASET_NAMED_MIXTURES:
        print(f"错误: 未知的混合数据集名称 '{data_mix}'")
        print(f"可用的混合数据集: {list(DATASET_NAMED_MIXTURES.keys())}")
        return
    
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    
    # 去重 (dataset_name, robot_type)
    seen = set()
    datasets_to_analyze = []
    for dataset_name, weight, robot_type in mixture_spec:
        key = (dataset_name, robot_type)
        if key in seen:
            continue
        seen.add(key)
        datasets_to_analyze.append((dataset_name, robot_type, weight))
    
    print(f"混合数据集包含 {len(datasets_to_analyze)} 个唯一数据集\n")
    
    # 准备任务参数
    tasks = []
    for dataset_name, robot_type, weight in datasets_to_analyze:
        dataset_path = data_root_dir / dataset_name
        tasks.append((str(dataset_path), robot_type, verbose))
    
    # 并行分析数据集
    all_results = []
    weights_map = {d[0]: d[2] for d in datasets_to_analyze}  # dataset_name -> weight
    
    if num_workers > 1:
        # 多进程并行
        print(f"使用 {num_workers} 个进程并行分析...")
        
        # 使用 spawn 方法避免一些共享状态问题
        ctx = mp.get_context('spawn')
        
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(_analyze_dataset_fast, task): task for task in tasks}
            
            # 使用 tqdm 显示进度
            with tqdm(total=len(tasks), desc="分析数据集") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    dataset_name = Path(task[0]).name
                    try:
                        result = future.result()
                        result["weight"] = weights_map.get(result["dataset_name"], 1.0)
                        all_results.append(result)
                        
                        # 显示简要结果
                        if result["status"] == "success":
                            ratio = result["missing_language_ratio"] * 100
                            total = result["total_trajectories"]
                            pbar.set_postfix_str(f"{dataset_name}: {total} 轨迹, {ratio:.1f}% 缺失")
                        else:
                            pbar.set_postfix_str(f"{dataset_name}: {result['status']}")
                    except Exception as e:
                        print(f"\n错误处理 {dataset_name}: {e}")
                        all_results.append({
                            "dataset_name": dataset_name,
                            "robot_type": task[1],
                            "status": "error",
                            "error_message": str(e),
                            "total_trajectories": 0,
                            "missing_language_trajectories": 0,
                            "valid_trajectories": 0,
                            "read_error_trajectories": 0,
                            "missing_language_ratio": 0.0,
                            "missing_trajectory_ids": [],
                            "error_trajectory_ids": [],
                            "sample_missing_languages": [],
                            "weight": weights_map.get(dataset_name, 1.0),
                        })
                    pbar.update(1)
    else:
        # 单进程串行
        for idx, task in enumerate(tqdm(tasks, desc="分析数据集"), 1):
            dataset_name = Path(task[0]).name
            
            result = _analyze_dataset_fast(task)
            result["weight"] = weights_map.get(result["dataset_name"], 1.0)
            all_results.append(result)
            
            if result["status"] == "success":
                print(f"  [{idx}/{len(tasks)}] {dataset_name}: {result['total_trajectories']} 轨迹, "
                      f"{result['missing_language_ratio']*100:.2f}% 缺失")
    
    # 汇总统计
    summary = {
        "total_datasets": len(datasets_to_analyze),
        "datasets_with_missing_language": 0,
        "total_trajectories": 0,
        "total_missing_language": 0,
        "total_read_errors": 0,
        "total_valid": 0,
    }
    
    for result in all_results:
        if result["status"] == "success":
            summary["total_trajectories"] += result["total_trajectories"]
            summary["total_missing_language"] += result["missing_language_trajectories"]
            summary["total_read_errors"] += result["read_error_trajectories"]
            summary["total_valid"] += result["valid_trajectories"]
            
            if result["missing_language_trajectories"] > 0:
                summary["datasets_with_missing_language"] += 1
    
    # 打印汇总报告
    print(f"\n{'='*80}")
    print("汇总报告")
    print(f"{'='*80}")
    
    print(f"\n总体统计:")
    print(f"  - 数据集总数: {summary['total_datasets']}")
    print(f"  - 存在缺失语言的数据集: {summary['datasets_with_missing_language']}")
    print(f"  - 轨迹总数: {summary['total_trajectories']}")
    print(f"  - 缺失语言轨迹: {summary['total_missing_language']} ({summary['total_missing_language']/max(1,summary['total_trajectories'])*100:.2f}%)")
    print(f"  - 读取错误轨迹: {summary['total_read_errors']}")
    print(f"  - 有效轨迹: {summary['total_valid']}")
    
    print(f"\n各数据集详情:")
    print("-" * 100)
    print(f"{'数据集名称':<40} {'机器人类型':<20} {'总轨迹':<10} {'缺失语言':<10} {'比例':<10} {'状态':<10}")
    print("-" * 100)
    
    # 按缺失比例排序
    sorted_results = sorted(all_results, key=lambda x: x.get("missing_language_ratio", 0), reverse=True)
    
    for r in sorted_results:
        name = r["dataset_name"][:38]
        robot = r["robot_type"][:18]
        total = r.get("total_trajectories", 0)
        missing = r.get("missing_language_trajectories", 0)
        ratio = r.get("missing_language_ratio", 0) * 100
        status = r["status"]
        
        if status == "success":
            status_str = "✓" if missing == 0 else f"⚠ {ratio:.1f}%"
        else:
            status_str = f"✗ {status}"
        
        print(f"{name:<40} {robot:<20} {total:<10} {missing:<10} {ratio:>6.2f}%    {status_str}")
    
    print("-" * 100)
    
    # 分析建议
    print(f"\n分析建议:")
    if summary["total_missing_language"] == 0:
        print("  ✓ 所有数据集都有完整的语言标注，无需过滤")
    elif summary["total_missing_language"] / max(1, summary["total_trajectories"]) < 0.01:
        print("  ⚠ 缺失语言轨迹比例很低 (<1%)，过滤对训练影响较小")
        print("  建议: 保留当前的过滤逻辑")
    elif summary["total_missing_language"] / max(1, summary["total_trajectories"]) < 0.05:
        print("  ⚠ 缺失语言轨迹比例较低 (<5%)，但可能值得关注")
        print("  建议: 检查是否可以补充语言标注，或考虑使用默认语言")
    else:
        print("  ✗ 缺失语言轨迹比例较高 (>5%)，需要特别关注")
        print("  建议:")
        print("    1. 检查数据集是否正确配置了语言标注")
        print("    2. 考虑使用默认语言填充缺失值")
        print("    3. 或者对于这些数据集禁用语言条件训练")
    
    # 保存详细结果
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "summary": summary,
            "datasets": all_results,
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="统计数据集中缺失语言标注的轨迹数目（高性能版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 分析混合数据集（32 进程并行）
    python analyze_missing_language.py --data_root_dir /data/lerobot --data_mix lam --num_workers 32
    
    # 使用配置文件
    python analyze_missing_language.py --config config/test.yaml --num_workers 16
    
    # 分析单个数据集
    python analyze_missing_language.py --data_root_dir /data/lerobot --dataset_name libero --robot_type FRANKA
        """
    )
    
    parser.add_argument("--config", type=str, help="YAML 配置文件路径")
    parser.add_argument("--data_root_dir", type=str, help="数据根目录")
    parser.add_argument("--data_mix", type=str, help="混合数据集名称")
    parser.add_argument("--dataset_name", type=str, help="单个数据集名称（可选）")
    parser.add_argument("--robot_type", type=str, help="机器人类型（配合 dataset_name 使用）")
    parser.add_argument("--video_backend", type=str, default="torchvision_av", help="视频后端（此版本未使用）")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    parser.add_argument("--output", "-o", type=str, help="输出 JSON 文件路径")
    parser.add_argument(
        "--num_workers", "-j", type=int, default=None,
        help="并行进程数（默认: CPU 核心数的一半，最多 32）"
    )
    
    args = parser.parse_args()
    
    # 设置默认并行度
    if args.num_workers is None:
        args.num_workers = min(32, max(1, cpu_count() // 2))
    
    # 从配置文件加载参数
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"错误: 配置文件不存在: {config_path}")
            sys.exit(1)
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        data_config = config.get("data", {})
        
        if not args.data_root_dir and "data_root_dir" in data_config:
            args.data_root_dir = data_config["data_root_dir"]
        if not args.data_mix and "data_mix" in data_config:
            args.data_mix = data_config["data_mix"]
        if "video_backend" in data_config:
            args.video_backend = data_config.get("video_backend", args.video_backend)
    
    # 验证参数
    if not args.data_root_dir:
        print("错误: 必须指定 --data_root_dir 或通过 --config 提供")
        sys.exit(1)
    
    data_root_dir = Path(args.data_root_dir)
    
    if args.dataset_name and args.robot_type:
        # 分析单个数据集
        dataset_path = data_root_dir / args.dataset_name
        result = analyze_single_dataset(
            dataset_path=dataset_path,
            robot_type=args.robot_type,
            video_backend=args.video_backend,
            verbose=args.verbose,
            num_workers=args.num_workers,
        )
        
        print(f"\n{'='*60}")
        print(f"数据集: {result['dataset_name']}")
        print(f"{'='*60}")
        print(f"总轨迹数: {result['total_trajectories']}")
        print(f"缺失语言: {result['missing_language_trajectories']} ({result['missing_language_ratio']*100:.2f}%)")
        print(f"读取错误: {result['read_error_trajectories']}")
        print(f"有效轨迹: {result['valid_trajectories']}")
        
        if result["sample_missing_languages"]:
            print(f"\n缺失语言轨迹示例:")
            for sample in result["sample_missing_languages"][:5]:
                print(f"  - trajectory_id={sample['trajectory_id']}, task_index={sample['task_index']}")
        
        if result.get("error_message"):
            print(f"\n错误信息: {result['error_message']}")
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {args.output}")
    
    elif args.data_mix:
        # 分析混合数据集
        analyze_mixture(
            data_root_dir=data_root_dir,
            data_mix=args.data_mix,
            video_backend=args.video_backend,
            verbose=args.verbose,
            output_file=args.output,
            num_workers=args.num_workers,
        )
    
    else:
        print("错误: 必须指定 --data_mix 或 (--dataset_name 和 --robot_type)")
        sys.exit(1)


if __name__ == "__main__":
    main()
