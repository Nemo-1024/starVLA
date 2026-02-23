from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.latent_world import (
    LatentWorldBatchBuilder,
    LatentWorldConfigBuilder,
    LatentWorldPromptBuilder,
    LatentWorldQwenVLInterface,
)
from starVLA.model.framework.vlas.latent_world_vla_independent import LatentWorldVLA
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("LatentWorldVLAIndependent")
@FRAMEWORK_REGISTRY.register("latent_world_vla_independent")
class LatentWorldVLAIndependentFramework(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        del kwargs

        self.config = config
        self.model_cfg = LatentWorldConfigBuilder(config).build()
        self.world_model, self.processor = LatentWorldVLA.build(self.model_cfg)

        self.qwen_vl_interface = LatentWorldQwenVLInterface(
            model=self.world_model.vlm,
            processor=self.world_model.processor,
            config=self.config,
        )
        self.action_model = self.world_model.flow

        prompt_builder = LatentWorldPromptBuilder(
            placeholder_token=self.model_cfg.latent_action_placeholder_token,
            act_queries=int(self.world_model.num_action_queries),
            flow_queries=int(self.world_model.flow.flow_action_query.shape[0]),
        )
        self.batch_builder = LatentWorldBatchBuilder(
            model_cfg=self.model_cfg,
            world_model=self.world_model,
            qwen_vl_interface=self.qwen_vl_interface,
            prompt_builder=prompt_builder,
            lam_image_hw=(256, 256),
        )

    def forward(self, examples: Sequence[dict], **kwargs) -> Dict[str, torch.Tensor]:
        del kwargs
        batch = self.batch_builder.build(examples, require_actions=True)
        out = self.world_model.forward(**batch)
        return {
            "total_loss": out["loss_total"],
            "loss_flow": out["loss_flow"],
            "loss_perceptual": out["loss_perceptual"],
            "loss_distill": out["loss_distill"],
            "loss_vlm": out["loss_vlm"],
        }

    @torch.inference_mode()
    def predict_action(self, examples: Sequence[dict], **kwargs) -> Dict[str, np.ndarray]:
        del kwargs
        batch = self.batch_builder.build(examples, require_actions=False)
        actions = self.world_model.predict_action(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            lam_videos=batch["lam_videos"],
            act_placeholder_mask=batch["act_placeholder_mask"],
            flow_placeholder_mask=batch["flow_placeholder_mask"],
            state=batch["state"],
            embodiment_id=batch["embodiment_id"],
            image_grid_thw=batch["image_grid_thw"],
            wrist_videos=batch["wrist_videos"],
        )
        if isinstance(actions, tuple):
            actions = actions[0]
        return {"normalized_actions": actions.detach().cpu().numpy()}

    def save_pretrained(self, save_directory: str, **kwargs):
        return self.world_model.save_pretrained(save_directory, **kwargs)
