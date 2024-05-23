from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class BasePruningSchedulerConfig:
    name: str = MISSING
    start: float = MISSING
    end: float = MISSING
    step: float = MISSING


@dataclass
class ManualSchedulerConfig(BasePruningSchedulerConfig):
    name: str = "manual"
    start: float = -1.0
    end: float = -1.0
    step: float = -1.0
    pruning_steps: list[list[float]] = MISSING
    # [[0.0, 0.1, 0.2, 0.3], [0.0, 0.1, 0.2, 0.3]] -> 2 pruning iterations with values per layer


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
