from dataclasses import dataclass, field
from typing import Any, Optional
from omegaconf import MISSING


@dataclass
class EarlyStopperConfig:
    enabled: bool = False
    monitor: str = "accuracy"
    patience: Any = MISSING
    mode: str = "max"
    min_delta: float = 0.0


@dataclass
class RestoreCheckpointConfig:
    enabled: bool = True
    monitor: str = "accuracy"
    mode: str = "max"
    min_delta: float = 0.0


@dataclass
class TrainerConfig:
    epochs: int = 10
    epochs_per_validation: int = 1
    early_stopper_cfg: EarlyStopperConfig = EarlyStopperConfig()
    restore_checkpoint_cfg: RestoreCheckpointConfig = RestoreCheckpointConfig()
