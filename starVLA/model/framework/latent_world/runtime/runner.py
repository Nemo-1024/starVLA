from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Sequence, Tuple

import numpy as np
import torch

from starVLA.model.framework.latent_world.types import (
    LatentWorldPolicyInferBatch,
    LatentWorldPolicyInferExample,
    LatentWorldPolicyTrainExample,
)

from .output_mapper import map_policy_infer_output, map_policy_train_output

if TYPE_CHECKING:
    from starVLA.model.framework.latent_world.batch_builder import (
        LatentWorldPolicyInferBatchBuilder,
        LatentWorldPolicyTrainBatchBuilder,
    )
    from starVLA.model.framework.vlas.latent_world_vla_independent import LatentWorldPolicyBackend


class LatentWorldPolicyRunner:
    def __init__(
        self,
        *,
        policy_backend: "LatentWorldPolicyBackend",
        train_batch_builder: "LatentWorldPolicyTrainBatchBuilder",
        infer_batch_builder: "LatentWorldPolicyInferBatchBuilder",
    ) -> None:
        self.policy_backend = policy_backend
        self.train_batch_builder = train_batch_builder
        self.infer_batch_builder = infer_batch_builder

    def train_step(self, examples: Sequence[LatentWorldPolicyTrainExample]) -> Dict[str, torch.Tensor]:
        batch = self.train_batch_builder.build_train_batch(examples)
        policy_output = self.policy_backend.forward(batch=batch)
        return map_policy_train_output(policy_output)

    @torch.inference_mode()
    def infer_step(self, examples: Sequence[LatentWorldPolicyInferExample]) -> Dict[str, np.ndarray]:
        batch = self.infer_batch_builder.build_infer_batch(examples)
        actions = self.policy_backend.predict_action(batch=batch)
        if isinstance(actions, tuple):
            actions = actions[0]
        return map_policy_infer_output(actions)

    @torch.inference_mode()
    def infer_step_with_aligned_targets(
        self,
        examples: Sequence[LatentWorldPolicyTrainExample],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build one aligned train batch, derive an infer batch view for action prediction,
        and return aligned targets from the train batch.

        Returns:
            pred_actions: [B, T, D]
            target_actions: [B, T, D] (already padded/aligned by train batch builder)
            action_mask: [B, T, D] boolean validity mask
        """
        train_batch = self.train_batch_builder.build_train_batch(examples)
        primary_video = train_batch["primary_video"]
        if primary_video.ndim != 5 or int(primary_video.shape[1]) < 1:
            raise ValueError(
                "Expected `primary_video` tensor with shape [B, T, C, H, W] and T>=1 "
                f"for aligned inference, got {tuple(primary_video.shape)}."
            )

        infer_batch: LatentWorldPolicyInferBatch = {
            "pixel_values": train_batch["pixel_values"],
            "input_ids": train_batch["input_ids"],
            "attention_mask": train_batch["attention_mask"],
            "act_placeholder_mask": train_batch["act_placeholder_mask"],
            "flow_placeholder_mask": train_batch["flow_placeholder_mask"],
            "primary_image": primary_video[:, 0, :, :, :],
            "state": train_batch["state"],
            "state_mask": train_batch["state_mask"],
            "embodiment_id": train_batch["embodiment_id"],
            "action_hz": train_batch["action_hz"],
            "image_grid_thw": train_batch["image_grid_thw"],
        }

        pred_actions = self.policy_backend.predict_action(batch=infer_batch)
        if isinstance(pred_actions, tuple):
            pred_actions = pred_actions[0]
        if not torch.is_tensor(pred_actions):
            pred_actions = torch.as_tensor(pred_actions)

        target_actions = train_batch["actions"]
        action_mask = train_batch["actions_mask"]
        if not torch.is_tensor(target_actions) or not torch.is_tensor(action_mask):
            raise TypeError("LatentWorld eval expects tensor `actions` and `actions_mask` from train_batch.")

        pred_actions = pred_actions.detach()
        target_actions = target_actions.to(
            device=pred_actions.device,
            dtype=pred_actions.dtype,
            non_blocking=True,
        )
        action_mask = action_mask.to(device=pred_actions.device, dtype=torch.bool, non_blocking=True)
        return pred_actions, target_actions, action_mask
