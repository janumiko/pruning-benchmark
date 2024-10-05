from dataclasses import dataclass

from architecture.pruners.structured_magnitude_pruner import StructuredMagnitudePruner
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch_pruning.pruner.importance import GroupNormImportance

from .schedulers import BasePruningScheduleConfig


@dataclass
class BaseImportanceConfig:
    _target_: str = MISSING


@dataclass
class NormImportanceConfig(BaseImportanceConfig):
    _target_: str = f"{GroupNormImportance.__module__}.{GroupNormImportance.__qualname__}"
    p: int = 2


@dataclass
class PrunerConfig:
    _target_: str = MISSING
    pruning_scheduler: BasePruningScheduleConfig = MISSING
    steps: int = MISSING
    pruning_config: dict = MISSING


@dataclass
class StructuredMagnitudePrunerConfig(PrunerConfig):
    _target_: str = (
        f"{StructuredMagnitudePruner.__module__}.{StructuredMagnitudePruner.__qualname__}"
    )
    importance: BaseImportanceConfig = MISSING


config_store = ConfigStore.instance()
config_store.store(group="pruner.importance", name="norm_importance", node=NormImportanceConfig)
config_store.store(
    group="pruner", name="structured_magnitude", node=StructuredMagnitudePrunerConfig
)
