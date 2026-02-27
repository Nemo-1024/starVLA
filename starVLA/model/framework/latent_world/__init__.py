from .batch_builder import LatentWorldPolicyBatchBuilder
from .runtime import LatentWorldPolicyComponents, build_policy_components
from .types import LiberoExample, LatentWorldPolicyInferBatch, LatentWorldPolicyTrainBatch
from .vlm_adapter import LatentWorldPolicyVLMAdapter


def __getattr__(name: str):
    if name == "LatentWorldPolicyConfigBuilder":
        from .config_builder import LatentWorldPolicyConfigBuilder

        return LatentWorldPolicyConfigBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LiberoExample",
    "LatentWorldPolicyBatchBuilder",
    "LatentWorldPolicyComponents",
    "LatentWorldPolicyConfigBuilder",
    "LatentWorldPolicyInferBatch",
    "LatentWorldPolicyTrainBatch",
    "LatentWorldPolicyVLMAdapter",
    "build_policy_components",
]
