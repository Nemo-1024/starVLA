from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.latent_world import LiberoExample, build_policy_components
from starVLA.model.framework.latent_world.runtime.freeze_policy import (
    apply_policy_freeze,
    parse_policy_freeze_config,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("LatentWorldVLAIndependent")
@FRAMEWORK_REGISTRY.register("latent_world_vla_independent")
class LatentWorldVLAIndependentFramework(baseframework):
    def __init__(self, config: Any = None, **kwargs) -> None:
        super().__init__()
        del kwargs

        self.config = config
        components = build_policy_components(config)

        self.policy_cfg = components.policy_cfg
        self.policy_backend = components.policy_backend
        self.policy_vlm_adapter = components.policy_vlm_adapter
        self.policy_batch_builder = components.policy_batch_builder
        self.policy_runner = components.runner
        self.policy_action_head = self.policy_backend.flow

        self.processor = self.policy_backend.processor

    def apply_training_freeze_policy(self, freeze_cfg) -> None:
        freeze_policy = parse_policy_freeze_config(freeze_cfg)
        apply_policy_freeze(self.policy_backend, freeze_policy)

    def forward(self, examples: Sequence[LiberoExample], **kwargs) -> Dict[str, torch.Tensor]:
        del kwargs
        return self.policy_runner.train_step(examples)

    @torch.inference_mode()
    def predict_action(self, examples: Sequence[LiberoExample], **kwargs) -> Dict[str, np.ndarray]:
        del kwargs
        return self.policy_runner.infer_step(examples)

    def save_pretrained(self, save_directory: str, **kwargs):
        return self.policy_backend.save_pretrained(save_directory, **kwargs)
