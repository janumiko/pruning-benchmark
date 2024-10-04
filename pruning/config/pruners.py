from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class Importance:
    _target_: str = MISSING


@dataclass
class NormImportance(Importance):
    _target_: str = "torch_pruning.importance.GroupNormImportance"
    p: int = 2


@dataclass
class PrunerConfig:
    _target_: str = MISSING


@dataclass
class StructuredMagnitudePrunerConfig(PrunerConfig):
    defaults: list[Any] = field(
            default_factory=lambda: [
                "_self_",
                {"importance": "norm_importance"},
            ]
        )

    _target_: str = "architecture.pruners.StructuredMagnitudePruner"
    importance: Importance = NormImportance()


config_store = ConfigStore.instance()
config_store.store(group="pruner/importance", name="norm_importance", node=NormImportance)
config_store.store(group="pruner", name="structured_magnitude", node=StructuredMagnitudePrunerConfig)
