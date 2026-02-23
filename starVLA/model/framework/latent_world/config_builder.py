from __future__ import annotations

from dataclasses import fields
from typing import Any

from starVLA.model.framework.vlas.flowmatching_expert import ConditionalFlowMatchingConfig
from starVLA.model.framework.vlas.latent_world_vla_independent import LatentWorldVLAConfig


class LatentWorldConfigBuilder:
    """Build LatentWorldVLAConfig from global training config with strict field validation."""

    _DEPRECATED_FIELDS = {"model_id", "yaml_path"}

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def build(self) -> LatentWorldVLAConfig:
        model_cfg = LatentWorldVLAConfig()
        lw = self.cfg.framework.latent_world

        top_keys = set(lw.keys())
        deprecated = sorted(self._DEPRECATED_FIELDS.intersection(top_keys))
        if deprecated:
            raise ValueError(
                "Deprecated latent_world fields are not allowed: "
                f"{deprecated}. Use `framework.qwenvl.base_vlm` as VLM source."
            )

        valid_top_keys = {f.name for f in fields(LatentWorldVLAConfig)} - {"flow_cfg", "model_id"}
        unknown_top = sorted(k for k in top_keys if k not in valid_top_keys and k != "flow_cfg")
        if unknown_top:
            raise ValueError(f"Unknown `framework.latent_world` fields: {unknown_top}")

        for key in top_keys:
            if key != "flow_cfg":
                setattr(model_cfg, key, lw[key])

        if "flow_cfg" in top_keys and lw.flow_cfg is not None:
            flow_keys = set(lw.flow_cfg.keys())
            legacy_flow_keys = sorted(k for k in ("proprio_dim", "use_proprio") if k in flow_keys)
            if legacy_flow_keys:
                raise ValueError(
                    "Legacy flow_cfg fields are not supported: "
                    f"{legacy_flow_keys}. Use `state_dim` / `use_state`."
                )
            valid_flow_keys = {f.name for f in fields(ConditionalFlowMatchingConfig)}
            unknown_flow = sorted(k for k in flow_keys if k not in valid_flow_keys)
            if unknown_flow:
                raise ValueError(f"Unknown `framework.latent_world.flow_cfg` fields: {unknown_flow}")
            for key in flow_keys:
                setattr(model_cfg.flow_cfg, key, lw.flow_cfg[key])

        model_cfg.model_id = self.cfg.framework.qwenvl.base_vlm
        return model_cfg
