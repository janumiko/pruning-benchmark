from dataclasses import dataclass

from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class BasePruningSchedulerConfig:
    _target_: str = MISSING
    steps: int = MISSING
    start: float = MISSING


@dataclass
class OneShotStepSchedulerConfig(BasePruningSchedulerConfig):
    _target_: str = "architecture.pruning.schedulers.OneShotStepScheduler"
    steps: int = 1
    start: float = 0.0


config_store = ConfigStore.instance()
config_store.store(group="pruner.pruning_scheduler", name="one_shot", node=OneShotStepSchedulerConfig)
