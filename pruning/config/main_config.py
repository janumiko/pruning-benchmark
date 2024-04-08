from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .datasets import CIFAR10, CIFAR100, BaseDataset, ImageNet1K
from .iterators import Iterative, OneShot, PruningIterator
from .optimizers import SGD, AdamW, BaseOptimizer


@dataclass
class Pruning:
    iterator: PruningIterator = MISSING
    finetune_epochs: int = MISSING
    _checkpoints_interval: tuple[float, float] = (0.7, 1.0)


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
            {"pruning.iterator": "_"},
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

# pruning iterators
config_store.store(group="pruning.iterator", name="iterative", node=Iterative)
config_store.store(group="pruning.iterator", name="one-shot", node=OneShot)
