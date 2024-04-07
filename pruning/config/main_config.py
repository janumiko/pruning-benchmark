from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .datasets import CIFAR10, CIFAR100, BaseDataset, ImageNet1K
from .optimizers import SGD, AdamW, BaseOptimizer


@dataclass
class Pruning:
    step_percent: int = MISSING
    iterations: int = MISSING
    finetune_epochs: int = MISSING
    _pruning_checkpoints: tuple[float] = tuple(range(60, 100, 2))


@dataclass
class Wandb:
    logging: bool = False
    project: Optional[str] = MISSING
    entity: Optional[str] = None
    job_type: Optional[str] = None


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
    enabled: bool = False
    patience: int = MISSING
    min_delta: float = 0.001


@dataclass
class MainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"optimizer": "_"},
            {"dataset": "_"},
        ]
    )

    dataset: BaseDataset = MISSING
    model: str = MISSING
    _checkpoint_path: str = MISSING
    optimizer: BaseOptimizer = MISSING

    pruning: Pruning = field(default_factory=Pruning)
    early_stopper: EarlyStopper = field(default_factory=EarlyStopper)
    dataloaders: Dataloaders = field(default_factory=Dataloaders)

    _repeat: int = 1
    _save_checkpoints: bool = False
    _seed: Seed = field(default_factory=Seed)
    _wandb: Wandb = field(default_factory=Wandb)


# register the config groups
config_store = ConfigStore.instance()
config_store.store(name="main_config", node=MainConfig)

# optimizers
config_store.store(group="optimizer", name="adamw", node=AdamW)
config_store.store(group="optimizer", name="sgd", node=SGD)

# datasets
config_store.store(group="dataset", name="cifar10", node=CIFAR10)
config_store.store(group="dataset", name="cifar100", node=CIFAR100)
config_store.store(group="dataset", name="imagenet1k", node=ImageNet1K)
