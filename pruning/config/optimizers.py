from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch.optim import SGD, Adam, AdamW


@dataclass
class BaseOptimizerConfig:
    _target_: str = MISSING
    lr: float = MISSING
    weight_decay: float = MISSING


@dataclass
class AdamOptimizerConfig(BaseOptimizerConfig):
    _target_: str = f"{Adam.__module__}.{Adam.__qualname__}"
    lr: float = MISSING
    weight_decay: float = 0.01


@dataclass
class AdamWOptimizerConfig(BaseOptimizerConfig):
    _target_: str = f"{AdamW.__module__}.{AdamW.__qualname__}"
    weight_decay: float = 0.01


@dataclass
class SGDOptimizerConfig(BaseOptimizerConfig):
    _target_: str = f"{SGD.__module__}.{SGD.__qualname__}"
    momentum: float = 0.9
    weight_decay: float = 0.0005


config_store = ConfigStore.instance()
config_store.store(group="optimizer", name="adam", node=AdamOptimizerConfig)
config_store.store(group="optimizer", name="adamw", node=AdamWOptimizerConfig)
config_store.store(group="optimizer", name="sgd", node=SGDOptimizerConfig)
