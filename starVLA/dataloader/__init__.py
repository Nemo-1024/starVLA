import json
import os
from pathlib import Path

import numpy as np
import torch.distributed as dist
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here
    # Hard fail if legacy VLM CoTrain config is present (VLM CoTrain has been removed).
    if OmegaConf.is_config(cfg) and hasattr(cfg, "datasets") and hasattr(cfg.datasets, "vlm_data"):
        raise ValueError(
            "VLM CoTrain has been removed. Remove 'datasets.vlm_data' from your config and use VLA-only training."
        )
    if dataset_py == "vlm_datasets":
        raise ValueError(
            "VLM CoTrain has been removed. Use dataset_py='lerobot_datasets' (or other VLA dataset) instead of 'vlm_datasets'."
        )

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=4,
            # shuffle=True
        )        
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
