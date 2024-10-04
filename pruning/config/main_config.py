from dataclasses import dataclass, field
from typing import Any, Optional
import uuid
import random

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .datasets import BaseDataset
from .optimizers import BaseOptimizer
from .trainer import TrainerConfig


@dataclass
class Dataloaders:
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = False


@dataclass
class DistributedConfig:
    enabled: bool = False
    init_method: str = MISSING
    world_size: int = MISSING


@dataclass
class ModelConfig:
    name: str = MISSING
    checkpoint_path: str = MISSING


@dataclass
class MainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"paths": "default.yaml"},
            {"hydra": "default.yaml"},
            {"optimizer": "_"},
            {"dataset": "_"},
        ]
    )
    paths: dict = field(default_factory=lambda: {})
    seed: int = random.randint(0, 1e6)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    group_id: Optional[str] = None

    distributed: DistributedConfig = DistributedConfig()
    trainer: TrainerConfig = TrainerConfig()

    model: ModelConfig = ModelConfig()
    optimizer: BaseOptimizer = MISSING
    dataset: BaseDataset = MISSING

    train_dataloader: Dataloaders = Dataloaders(drop_last=True)
    validation_dataloader: Dataloaders = Dataloaders(drop_last=False)


# register the config groups
config_store = ConfigStore.instance()
config_store.store(name="main_config", node=MainConfig)
