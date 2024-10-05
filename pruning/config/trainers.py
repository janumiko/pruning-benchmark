from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class EarlyStopperConfig:
    _target_: str = "architecture.utils.training_utils.EarlyStopper"
    enabled: bool = False
    monitor: str = "accuracy"
    patience: Any = MISSING
    mode: str = "max"
    min_delta: float = 0.0
    overide_epochs_to_inf: bool = False


@dataclass
class RestoreCheckpointConfig:
    _target_: str = "architecture.utils.training_utils.RestoreCheckpoint"
    enabled: bool = True
    monitor: str = "accuracy"
    mode: str = "max"
    min_delta: float = 0.0


@dataclass
class TrainerConfig:
    _target_: str = MISSING
    epochs: int = MISSING
    epochs_per_validation: int = 1
    early_stopper: EarlyStopperConfig = EarlyStopperConfig()
    restore_checkpoint: RestoreCheckpointConfig = RestoreCheckpointConfig()


@dataclass
class ClassificationTrainerConfig(TrainerConfig):
    _target_: str = "architecture.trainers.classification_trainer.ClassificationTrainer"
    epochs: int = 10


config_store = ConfigStore.instance()
config_store.store(group="trainer", name="classification", node=ClassificationTrainerConfig)
