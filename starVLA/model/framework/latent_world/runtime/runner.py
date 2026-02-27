from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Sequence

import numpy as np
import torch

from starVLA.model.framework.latent_world.types import LiberoExample

from .output_mapper import map_policy_infer_output, map_policy_train_output

if TYPE_CHECKING:
    from starVLA.model.framework.latent_world.batch_builder import LatentWorldPolicyBatchBuilder
    from starVLA.model.framework.vlas.latent_world_vla_independent import LatentWorldPolicyBackend


class LatentWorldPolicyRunner:
    def __init__(
        self,
        *,
        policy_backend: "LatentWorldPolicyBackend",
        policy_batch_builder: "LatentWorldPolicyBatchBuilder",
    ) -> None:
        self.policy_backend = policy_backend
        self.policy_batch_builder = policy_batch_builder

    def train_step(self, examples: Sequence[LiberoExample]) -> Dict[str, torch.Tensor]:
        batch = self.policy_batch_builder.build_train_batch(examples)
        policy_output = self.policy_backend.forward(batch=batch)
        return map_policy_train_output(policy_output)

    @torch.inference_mode()
    def infer_step(self, examples: Sequence[LiberoExample]) -> Dict[str, np.ndarray]:
        batch = self.policy_batch_builder.build_infer_batch(examples)
        actions = self.policy_backend.predict_action(batch=batch)
        if isinstance(actions, tuple):
            actions = actions[0]
        return map_policy_infer_output(actions)
