from dataclasses import dataclass

from architecture.pruners.structured_pruner import StructuredPruner
from architecture.pruners.unstructured_pruner import UnstructuredPruner
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch_pruning.pruner.importance import GroupNormImportance, GroupHessianImportance

from .schedulers import BasePruningScheduleConfig


@dataclass
class BaseImportanceConfig:
    _target_: str = MISSING


@dataclass
class NormImportanceConfig(BaseImportanceConfig):
    _target_: str = f"{GroupNormImportance.__module__}.{GroupNormImportance.__qualname__}"
    p: int = 2


@dataclass
class HessianImportanceConfig(BaseImportanceConfig):
    _target_: str = f"{GroupHessianImportance.__module__}.{GroupHessianImportance.__qualname__}"
    group_reduction: str = "mean"
    normalizer: str = "mean"
    bias = False


@dataclass
class PrunerConfig:
    _target_: str = MISSING
    pruning_scheduler: BasePruningScheduleConfig = MISSING
    steps: int = MISSING
    pruning_config: dict = MISSING


@dataclass
class StructuredPrunerConfig(PrunerConfig):
    _target_: str = f"{StructuredPruner.__module__}.{StructuredPruner.__qualname__}"
    importance: BaseImportanceConfig = MISSING
    global_pruning: bool = False


@dataclass
class UnstructuredPrunerConfig(PrunerConfig):
    _target_: str = f"{UnstructuredPruner.__module__}.{UnstructuredPruner.__qualname__}"
    pruning_ratio: float = MISSING
    global_pruning: bool = True


config_store = ConfigStore.instance()
config_store.store(group="pruner.importance", name="norm_importance", node=NormImportanceConfig)
config_store.store(group="pruner.importance", name="hessian_importance", node=HessianImportanceConfig)
config_store.store(group="pruner", name="structured", node=StructuredPrunerConfig)
config_store.store(group="pruner", name="unstructured", node=UnstructuredPrunerConfig)
