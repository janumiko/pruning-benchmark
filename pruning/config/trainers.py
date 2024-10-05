from dataclasses import dataclass, field
from typing import Any

from architecture.utils.training_utils import EarlyStopper, RestoreCheckpoint
from architecture.trainers.classification_trainer import ClassificationTrainer
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class EarlyStopperConfig:
    _target_: str = f"{EarlyStopper.__module__}.{EarlyStopper.__qualname__}"
    enabled: bool = False
    monitor: str = "accuracy"
    patience: Any = MISSING
    mode: str = "max"
    min_delta: float = 0.0
    override_epochs_to_inf: bool = False


@dataclass
class RestoreCheckpointConfig:
    _target_: str = f"{RestoreCheckpoint.__module__}.{RestoreCheckpoint.__qualname__}"
    enabled: bool = True
    monitor: str = "accuracy"
    mode: str = "max"
    min_delta: float = 0.0


@dataclass
class TrainerConfig:
    _target_: str = MISSING
    epochs: int = MISSING
    epochs_per_validation: int = 1
    early_stopper: EarlyStopperConfig = field(default_factory=EarlyStopperConfig)
    restore_checkpoint: RestoreCheckpointConfig = field(default_factory=RestoreCheckpointConfig)


@dataclass
class ClassificationTrainerConfig(TrainerConfig):
    _target_: str = f"{ClassificationTrainer.__module__}.{ClassificationTrainer.__qualname__}"
    epochs: int = 10


config_store = ConfigStore.instance()
config_store.store(group="trainer", name="classification", node=ClassificationTrainerConfig)
