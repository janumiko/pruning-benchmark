from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class BasePruningSchedulerConfig:
    name: str = MISSING
    start: float = MISSING
    end: float = MISSING
    step: float = MISSING


@dataclass
class IterativeStepSchedulerConfig(BasePruningSchedulerConfig):
    name: str = "iterative"
    start: float = 0.0


@dataclass
class OneShotStepSchedulerConfig(BasePruningSchedulerConfig):
    name: str = "one-shot"
    start: float = 0.0
    step: float = 0.0


@dataclass
class ConstantStepSchedulerConfig(BasePruningSchedulerConfig):
    name: str = "constant"
    start: float = 0.0


@dataclass
class LogarithmicStepSchedulerConfig(BasePruningSchedulerConfig):
    name: str = "logarithmic"
    start: float = 0.0
