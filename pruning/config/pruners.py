from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from .schedulers import BasePruningSchedulerConfig, OneShotStepSchedulerConfig


@dataclass
class BaseImportanceConfig:
    _target_: str = MISSING


@dataclass
class NormImportanceConfig(BaseImportanceConfig):
    _target_: str = "torch_pruning.importance.GroupNormImportance"
    p: int = 2


@dataclass
class PrunerConfig:
    _target_: str = MISSING
    pruning_scheduler: BasePruningSchedulerConfig = MISSING


@dataclass
class StructuredMagnitudePrunerConfig(PrunerConfig):
    _target_: str = "architecture.pruners.StructuredMagnitudePruner"
    importance: BaseImportanceConfig = MISSING


config_store = ConfigStore.instance()
config_store.store(group="pruner.importance", name="norm_importance", node=NormImportanceConfig)
config_store.store(
    group="pruner", name="structured_magnitude", node=StructuredMagnitudePrunerConfig
)
