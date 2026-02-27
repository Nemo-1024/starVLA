from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def map_policy_train_output(policy_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "total_loss": policy_output["loss_total"],
        "loss_flow": policy_output["loss_flow"],
        "loss_perceptual": policy_output["loss_perceptual"],
        "loss_distill": policy_output["loss_distill"],
        "loss_vlm": policy_output["loss_vlm"],
    }


def map_policy_infer_output(actions: torch.Tensor) -> Dict[str, np.ndarray]:
    return {"normalized_actions": actions.detach().cpu().numpy()}
