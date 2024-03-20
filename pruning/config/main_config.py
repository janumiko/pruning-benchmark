from dataclasses import dataclass, field
from typing import Any, Optional
from .optimizers import AdamW, SGD, BaseOptimizer
from .models import BaseModel, ResNet18_CIFAR10, ResNet18_CIFAR100, LeNet_CIFAR10
from .datasets import BaseDataset, CIFAR10, CIFAR100
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class Pruning:
    iteration_rate: float = MISSING
    iterations: int = MISSING
    finetune_epochs: int = MISSING
    early_stopping: bool = False


@dataclass
class Wandb:
    logging: bool = False
    project: Optional[str] = MISSING


@dataclass
class Seed:
    is_set: bool = False
    value: Optional[int] = MISSING


@dataclass
class Dataloaders:
    _pin_memory: bool = True
    _num_workers: int = 4
    batch_size: int = 128


@dataclass
class EarlyStopper:
    patience: int = MISSING
    min_delta: float = 0.001


@dataclass
class MainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            {"optimizer": "_"},
            {"model": "_"},
            {"dataset": "_"},
        ]
    )

    dataset: BaseDataset = MISSING
    model: BaseModel = MISSING
    optimizer: BaseOptimizer = MISSING

    pruning: Pruning = field(default_factory=Pruning)
    early_stopper: EarlyStopper = field(default_factory=EarlyStopper)
    dataloaders: Dataloaders = field(default_factory=Dataloaders)

    _repeat: int = 1
    _save_checkpoints: bool = False
    _seed: Seed = field(default_factory=Seed)
    _wandb: Wandb = field(default_factory=Wandb)


config_store = ConfigStore.instance()
config_store.store(name="main_config", node=MainConfig)
config_store.store(group="optimizer", name="adamw", node=AdamW)
config_store.store(group="optimizer", name="sgd", node=SGD)
config_store.store(group="model", name="resnet18_cifar10", node=ResNet18_CIFAR10)
config_store.store(group="model", name="resnet18_cifar100", node=ResNet18_CIFAR100)
config_store.store(group="model", name="lenet_cifar10", node=LeNet_CIFAR10)
config_store.store(group="dataset", name="cifar10", node=CIFAR10)
config_store.store(group="dataset", name="cifar100", node=CIFAR100)
