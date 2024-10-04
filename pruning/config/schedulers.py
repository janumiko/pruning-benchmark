from dataclasses import dataclass

from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class BasePruningSchedulerConfig:
    _target_: str = MISSING
    start: float = MISSING
    end: float = MISSING
    steps: int = MISSING


@dataclass
class OneShotStepSchedulerConfig(BasePruningSchedulerConfig):
    _target_: str = "architecture.pruning.schedulers.OneShotStepScheduler"
    steps: int = 1
    end: float = MISSING
    start: float = 0.0
