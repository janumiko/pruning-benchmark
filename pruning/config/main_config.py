from dataclasses import dataclass, field
from typing import Any, Optional
import uuid

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
    _checkpoints_interval: Interval = field(default_factory=lambda: Interval(0.5, 1.0))


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
class EarlyStopperConfig:
    enabled: bool = False
    patience: int = 5
    min_delta: float = 0
    metric: BaseMetric = field(
        default_factory=lambda: BaseMetric("validation_loss", is_decreasing=True)
    )


@dataclass
class DistributedConfig:
    enabled: bool = False
    init_method: str = MISSING
    world_size: int = MISSING


@dataclass
class MainConfig:
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"paths": "default.yaml"},
            {"hydra": "default.yaml"},
        ]
    )
    paths: dict = field(default_factory=lambda: {})
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    distributed: DistributedConfig = DistributedConfig()


# register the config groups
config_store = ConfigStore.instance()
config_store.store(name="main_config", node=MainConfig)
