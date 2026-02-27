from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from starVLA.model.framework.vlas.vlm_auto import (
    _resolve_llm_module,
    _unfreeze_last_n_llm_layers,
    freeze_qwen3vl,
)


@dataclass(frozen=True)
class LatentWorldPolicyFreezeConfig:
    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    freeze_last_llm_layer: bool = False
    freeze_embedding: bool = False
    unfreeze_vision_merger: bool = False
    unfreeze_lam_decoder: bool = False
    unfreeze_llm_last_n_layers: Optional[int] = None


def parse_policy_freeze_config(freeze_cfg: Any) -> LatentWorldPolicyFreezeConfig:
    if freeze_cfg is None:
        return LatentWorldPolicyFreezeConfig()

    unfreeze_last_n = freeze_cfg.get("unfreeze_llm_last_n_layers", None)
    if unfreeze_last_n is not None:
        unfreeze_last_n = int(unfreeze_last_n)

    return LatentWorldPolicyFreezeConfig(
        freeze_vision_backbone=bool(freeze_cfg.get("freeze_vision_backbone", False)),
        freeze_llm_backbone=bool(freeze_cfg.get("freeze_llm_backbone", False)),
        freeze_last_llm_layer=bool(freeze_cfg.get("freeze_last_llm_layer", False)),
        freeze_embedding=bool(freeze_cfg.get("freeze_embedding", False)),
        unfreeze_vision_merger=bool(freeze_cfg.get("unfreeze_vision_merger", False)),
        unfreeze_lam_decoder=bool(freeze_cfg.get("unfreeze_lam_decoder", False)),
        unfreeze_llm_last_n_layers=unfreeze_last_n,
    )


def apply_policy_freeze(
    policy_backend,
    freeze_policy: LatentWorldPolicyFreezeConfig,
) -> None:
    freeze_qwen3vl(
        policy_backend.vlm,
        freeze_vision_backbone=freeze_policy.freeze_vision_backbone,
        freeze_llm_backbone=freeze_policy.freeze_llm_backbone,
        freeze_last_llm_layer=freeze_policy.freeze_last_llm_layer,
        freeze_embedding=freeze_policy.freeze_embedding,
        unfreeze_vision_merger=freeze_policy.unfreeze_vision_merger,
    )

    if (
        freeze_policy.freeze_llm_backbone
        and freeze_policy.unfreeze_llm_last_n_layers is not None
        and freeze_policy.unfreeze_llm_last_n_layers > 0
    ):
        llm_module = _resolve_llm_module(policy_backend.vlm)
        if llm_module is not None:
            _unfreeze_last_n_llm_layers(llm_module, freeze_policy.unfreeze_llm_last_n_layers)

    for p in policy_backend.lam.parameters():
        p.requires_grad = False
    if freeze_policy.unfreeze_lam_decoder:
        lam_decoder = getattr(policy_backend.lam, "decoder", None)
        if lam_decoder is not None:
            for p in lam_decoder.parameters():
                p.requires_grad = True
