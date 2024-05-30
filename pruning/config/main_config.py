from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch import nn

from .datasets import CIFAR10, CIFAR100, BaseDataset, ImageNet1K
from .methods import (
    BasePruningMethodConfig,
    GlobalL1UnstructuredConfig,
    LnStructuredConfig,
)
from .metrics import BaseMetric, Top1Accuracy, Top5Accuracy, ValidationLoss
from .optimizers import SGD, AdamW, BaseOptimizer
from .schedulers import (
    BasePruningSchedulerConfig,
    ConstantStepSchedulerConfig,
    IterativeStepSchedulerConfig,
    LogarithmicStepSchedulerConfig,
    ManualSchedulerConfig,
    OneShotStepSchedulerConfig,
)

# TODO: add the pruning types to Hydra config
TYPES_TO_PRUNE: tuple[nn.Module] = (nn.Linear, nn.Conv2d)


@dataclass
class Interval:
    start: float
    end: float


@dataclass
class Pruning:
    scheduler: BasePruningSchedulerConfig = MISSING
    method: BasePruningMethodConfig = MISSING
    finetune_epochs: int = MISSING
    _checkpoints_interval: Interval = field(default_factory=lambda: Interval(0.0, 1.0))


@dataclass
class Wandb:
    logging: bool = False
    project: Optional[str] = MISSING
    entity: Optional[str] = None
    job_type: Optional[str] = None


@dataclass
class Dataloaders:
    _pin_memory: bool = True
    _drop_last: bool = False
    _persistent_workers: bool = False
    _num_workers: int = 8
    batch_size: int = 128


@dataclass
class EarlyStopperConfig:
    enabled: bool = False
    patiences: list[int] = field(default_factory=lambda: [0])
    min_delta: float = 0
    metric: BaseMetric = field(
        default_factory=lambda: BaseMetric("validation_loss", is_decreasing=True)
    )


@dataclass
class MainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"optimizer": "_"},
            {"dataset": "_"},
            {"pruning.scheduler": "_"},
            {"pruning.method": "_"},
            {"early_stopper.metric": "validation_loss"},
        ]
    )

    dataset: BaseDataset = MISSING
    model: str = MISSING
    _checkpoint_path: str = MISSING
    optimizer: BaseOptimizer = MISSING

    pruning: Pruning = field(default_factory=Pruning)
    early_stopper: EarlyStopperConfig = field(default_factory=EarlyStopperConfig)
    dataloaders: Dataloaders = field(default_factory=Dataloaders)
    best_checkpoint_criterion: BaseMetric = field(default_factory=Top1Accuracy)

    _gpus: int = 1
    _shared_filesystem: Optional[str] = None
    _repeat: int = 1
    _save_checkpoints: bool = False
    _seed: Optional[int] = None
    _wandb: Wandb = field(default_factory=Wandb)
    _logging_level: str = "INFO"  # DEBUG


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

# pruning schedulers
config_store.store(
    group="pruning.scheduler",
    name=IterativeStepSchedulerConfig().name,
    node=IterativeStepSchedulerConfig,
)
config_store.store(
    group="pruning.scheduler",
    name=OneShotStepSchedulerConfig().name,
    node=OneShotStepSchedulerConfig,
)
config_store.store(
    group="pruning.scheduler",
    name=LogarithmicStepSchedulerConfig().name,
    node=LogarithmicStepSchedulerConfig,
)
config_store.store(
    group="pruning.scheduler",
    name=ConstantStepSchedulerConfig().name,
    node=ConstantStepSchedulerConfig,
)
config_store.store(
    group="pruning.scheduler",
    name=ManualSchedulerConfig().name,
    node=ManualSchedulerConfig,
)

# pruning methods
config_store.store(group="pruning.method", name=LnStructuredConfig().name, node=LnStructuredConfig)
config_store.store(
    group="pruning.method", name=GlobalL1UnstructuredConfig().name, node=GlobalL1UnstructuredConfig
)

# metrics
config_store.store(group="early_stopper.metric", name=Top1Accuracy().name, node=Top1Accuracy)
config_store.store(group="early_stopper.metric", name=Top5Accuracy().name, node=Top5Accuracy)
config_store.store(group="early_stopper.metric", name=ValidationLoss().name, node=ValidationLoss)
