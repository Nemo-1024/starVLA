import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch.distributed as dist
from accelerate.logging import get_logger
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

def build_dataloaders(cfg) -> tuple[DataLoader, Optional[DataLoader]]:
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    vla_dataset_cfg = cfg.datasets.vla_data
    batch_size = cfg.datasets.vla_data.per_device_batch_size
    num_workers = 4

    vla_train_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg, mode="train")

    vla_train_dataloader = DataLoader(
        vla_train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        # shuffle=True
    )

    vla_val_dataloader = None
    try:
        vla_val_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg, mode="val")
        vla_val_dataloader = DataLoader(
            vla_val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            # shuffle=False
        )
    except ValueError as exc:
        logger.warning(f"Validation dataset unavailable, continue without val loader: {exc}")

    if (not dist.is_initialized()) or dist.get_rank() == 0:
        output_dir = Path(cfg.output_dir)
        vla_train_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
    return vla_train_dataloader, vla_val_dataloader


def build_dataloader(cfg):
    return build_dataloaders(cfg)[0]
