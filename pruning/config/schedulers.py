from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class BasePruningSchedulerConfig:
    name: str = MISSING


@dataclass
class GeometricStepSchedulerConfig(BasePruningSchedulerConfig):
    name: str = "geometric"


@dataclass
class OneShotStepSchedulerConfig(BasePruningSchedulerConfig):
    name: str = "one-shot"


@dataclass
class ConstantStepSchedulerConfig(BasePruningSchedulerConfig):
    name: str = "constant"


@dataclass
class FewShotScheduleConfig(BasePruningSchedulerConfig):
    name: str = "few-shot"
    start: float = 0.0
