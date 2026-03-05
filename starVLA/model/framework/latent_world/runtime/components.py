from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from starVLA.model.framework.latent_world.batch_builder import (
    LatentWorldPolicyInferBatchBuilder,
    LatentWorldPolicyTrainBatchBuilder,
)
from starVLA.model.framework.latent_world.config_builder import LatentWorldPolicyConfigBuilder
from starVLA.model.framework.latent_world.vlm_adapter import LatentWorldPolicyVLMAdapter
from starVLA.model.framework.vlas.latent_world_vla_independent import (
    LatentWorldPolicyBackend,
    LatentWorldPolicyConfig,
)

from .contracts import validate_policy_contract
from .runner import LatentWorldPolicyRunner


@dataclass
class LatentWorldPolicyComponents:
    policy_cfg: LatentWorldPolicyConfig
    policy_backend: LatentWorldPolicyBackend
    policy_vlm_adapter: LatentWorldPolicyVLMAdapter
    train_batch_builder: LatentWorldPolicyTrainBatchBuilder
    infer_batch_builder: LatentWorldPolicyInferBatchBuilder
    runner: LatentWorldPolicyRunner


def build_policy_components(config: Any) -> LatentWorldPolicyComponents:
    policy_cfg = LatentWorldPolicyConfigBuilder(config).build()
    validate_policy_contract(config, policy_cfg)

    vlm_model_id = config.framework.qwenvl.base_vlm
    if vlm_model_id is None:
        raise ValueError("Missing `framework.qwenvl.base_vlm` for LatentWorldVLAIndependent.")

    policy_backend = LatentWorldPolicyBackend.build(policy_cfg, vlm_model_id=str(vlm_model_id))
    policy_vlm_adapter = LatentWorldPolicyVLMAdapter(
        model=policy_backend.vlm,
        processor=policy_backend.processor,
        config=config,
        placeholder_token=policy_cfg.latent_action_placeholder_token,
        act_queries=int(policy_backend.num_action_queries),
        flow_queries=int(policy_backend.flow.flow_action_query.shape[0]),
    )
    train_batch_builder = LatentWorldPolicyTrainBatchBuilder(
        policy_cfg=policy_cfg,
        policy_backend=policy_backend,
        policy_vlm_adapter=policy_vlm_adapter,
        lam_image_hw=(256, 256),
    )
    infer_batch_builder = LatentWorldPolicyInferBatchBuilder(
        policy_cfg=policy_cfg,
        policy_backend=policy_backend,
        policy_vlm_adapter=policy_vlm_adapter,
        lam_image_hw=(256, 256),
    )
    runner = LatentWorldPolicyRunner(
        policy_backend=policy_backend,
        train_batch_builder=train_batch_builder,
        infer_batch_builder=infer_batch_builder,
    )

    return LatentWorldPolicyComponents(
        policy_cfg=policy_cfg,
        policy_backend=policy_backend,
        policy_vlm_adapter=policy_vlm_adapter,
        train_batch_builder=train_batch_builder,
        infer_batch_builder=infer_batch_builder,
        runner=runner,
    )
