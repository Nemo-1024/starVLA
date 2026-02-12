import os
from typing import Optional, Union
from functools import partial

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .collate import lam_collate
from .lerobot_dataset import LeRobotLAMDataset


def _lam_worker_init_fn(_worker_id: int) -> None:
    # Ensure each DataLoader worker stays single-threaded to avoid CPU oversubscription.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["BLIS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # It can only be set once in a process; ignore if already initialized.
        pass


class LeRobotDataModule(LightningDataModule):
    def __init__(
        self,
        data_root_dir: str,
        data_mix: str,
        num_frames: int = 5,
        video_backend: str = "torchvision_av",
        preferred_video_key: Optional[str] = None,
        state_keys: Optional[list[str]] = None,
        # Temporal sampling controls:
        # - If video_delta_indices/state_delta_indices are provided, they take precedence.
        # - Else if frame_dt_sec is provided, we convert dt->stride using dataset video fps at runtime.
        # - Otherwise we will use range(0, num_frames * frame_stride, frame_stride).
        frame_dt_sec: Optional[float] = None,
        frame_stride: int = 1,
        video_delta_indices: Optional[list[int]] = None,
        state_delta_indices: Optional[list[int]] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        prefetch_factor: Optional[int] = None,  # None = DataLoader default(2). 若 CPU 成瓶颈可调大(如 8) 或减小 num_workers(如 2)
        in_order: bool = False,  # False: 允许快 worker 先返回 batch，减少按 worker 数周期性等待
        debug_repeat_batch: Union[bool, int] = False,
        max_proprio_dim: int = 32,  # Maximum proprio dimension for padding
    ):
        super().__init__()
        self.data_root_dir = data_root_dir
        self.data_mix = data_mix
        self.num_frames = num_frames
        self.video_backend = video_backend
        self.preferred_video_key = preferred_video_key
        self.state_keys = state_keys
        self.frame_dt_sec = frame_dt_sec
        self.frame_stride = frame_stride
        self.video_delta_indices = video_delta_indices
        self.state_delta_indices = state_delta_indices
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.in_order = in_order
        self.debug_repeat_batch = debug_repeat_batch
        self.max_proprio_dim = max_proprio_dim

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = LeRobotLAMDataset(
                data_root_dir=self.data_root_dir,
                data_mix=self.data_mix,
                num_frames=self.num_frames,
                video_backend=self.video_backend,
                preferred_video_key=self.preferred_video_key,
                state_keys=self.state_keys,
                frame_dt_sec=self.frame_dt_sec,
                frame_stride=self.frame_stride,
                video_delta_indices=self.video_delta_indices,
                state_delta_indices=self.state_delta_indices,
                debug_repeat_batch=self.debug_repeat_batch,
            )
            # 简化：验证集复用训练数据
            # self.val_dataset = self.train_dataset

    def train_dataloader(self):
        # Use partial to pass max_proprio_dim to collate function
        collate_fn = partial(lam_collate, max_proprio_dim=self.max_proprio_dim)
        # DDP: Lightning 会自动注入 DistributedSampler，各 rank 得到不重叠的 index 子集，无需手写 sampler
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
            worker_init_fn=_lam_worker_init_fn if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            in_order=self.in_order if self.num_workers > 0 else True,
        )

    def on_train_epoch_start(self, trainer) -> None:
        """每 epoch 更新 mixture 的 epoch，使 sample_step 的 RNG 随 epoch 变化，便于复现与多样性。"""
        if self.train_dataset is not None and hasattr(self.train_dataset, "mixture"):
            mixture = getattr(self.train_dataset, "mixture", None)
            if mixture is not None and hasattr(mixture, "set_epoch"):
                mixture.set_epoch(trainer.current_epoch)

    # def val_dataloader(self):
    #     # Use partial to pass max_proprio_dim to collate function
    #     collate_fn = partial(lam_collate, max_proprio_dim=self.max_proprio_dim)
        
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #         pin_memory=True,
    #         collate_fn=collate_fn,
    #         persistent_workers=self.num_workers > 0,
    #         prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
    #     )
