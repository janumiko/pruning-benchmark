from dataclasses import dataclass, field
import random
from typing import Any, Optional
import uuid

from architecture.utils.metrics import WandbMetricLogger
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .datasets import BaseDataset
from .optimizers import BaseOptimizerConfig
from .pruner.pruners import PrunerConfig
from .trainers import TrainerConfig


@dataclass
class Dataloaders:
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last: bool = False


@dataclass
class WandbConfig:
    _target_: str = f"{WandbMetricLogger.__module__}.{WandbMetricLogger.__name__}"
    project: str = MISSING
    mode: str = "disabled"
    entity: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    group: Optional[str] = None
    job_type: Optional[str] = None


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
            {"wandb": "default.yaml"},
            {"optimizer": "_"},
            {"dataset": "_"},
            {"trainer": "_"},
            {"pruner": "_"},
            {"pruner.pruning_scheduler": "_"},
            {"pruner.importance": None},
            {"pruner/pruning_config": "default.yaml"},
        ]
    )
    paths: dict = field(default_factory=dict)
    seed: int = random.randint(0, 1e6)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    wandb: WandbConfig = field(default_factory=WandbConfig)

    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    trainer: TrainerConfig = MISSING
    pruner: PrunerConfig = MISSING

    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: BaseOptimizerConfig = MISSING
    dataset: BaseDataset = MISSING

    train_dataloader: Dataloaders = field(default_factory=Dataloaders)
    validation_dataloader: Dataloaders = field(default_factory=Dataloaders)


# register the config groups
config_store = ConfigStore.instance()
config_store.store(name="main_config", node=MainConfig)
