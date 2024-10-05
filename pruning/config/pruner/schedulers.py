from dataclasses import dataclass

from architecture.pruners.schedulers import (
    ConstantPruningSchedule,
    FewShotPruningSchedule,
    GeometricPruningSchedule,
    OneShotPruningSchedule,
)
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BasePruningScheduleConfig:
    _target_: str = MISSING
    rounding_precision: int = 8


@dataclass
class GeometricPruningScheduleConfig(BasePruningScheduleConfig):
    _target_: str = (
        f"{GeometricPruningSchedule.__module__}.{GeometricPruningSchedule.__qualname__}"
    )


@dataclass
class OneShotPruningScheduleConfig(BasePruningScheduleConfig):
    _target_: str = f"{OneShotPruningSchedule.__module__}.{OneShotPruningSchedule.__qualname__}"


@dataclass
class ConstantPruningScheduleConfig(BasePruningScheduleConfig):
    _target_: str = f"{ConstantPruningSchedule.__module__}.{ConstantPruningSchedule.__qualname__}"


@dataclass
class FewShotPruningScheduleConfig(BasePruningScheduleConfig):
    _target_: str = f"{FewShotPruningSchedule.__module__}.{FewShotPruningSchedule.__qualname__}"
    one_shot_step: float = MISSING


config_store = ConfigStore.instance()
config_store.store(
    group="pruner.pruning_scheduler", name="one_shot", node=OneShotPruningScheduleConfig
)
config_store.store(
    group="pruner.pruning_scheduler", name="constant", node=ConstantPruningScheduleConfig
)
config_store.store(
    group="pruner.pruning_scheduler", name="geometric", node=GeometricPruningScheduleConfig
)
config_store.store(
    group="pruner.pruning_scheduler", name="few_shot", node=FewShotPruningScheduleConfig
)
