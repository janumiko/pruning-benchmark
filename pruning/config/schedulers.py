from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BasePruningSchedulerConfig:
    _target_: str = MISSING
    steps: int = MISSING


@dataclass
class OneShotPruningSchedulerConfig(BasePruningSchedulerConfig):
    _target_: str = "architecture.pruning.schedulers.OneShotStepScheduler"
    steps: int = 1


config_store = ConfigStore.instance()
config_store.store(group="pruner.pruning_scheduler", name="one_shot", node=OneShotPruningSchedulerConfig)
