from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BaseOptimizer:
    _target_: str = MISSING
    lr: float = MISSING
    weight_decay: float = MISSING


@dataclass
class Adam(BaseOptimizer):
    _target_: str = "torch.optim.Adam"
    lr: float = 0.001
    weight_decay: float = 0.01


@dataclass
class AdamW(BaseOptimizer):
    _target_: str = "torch.optim.AdamW"
    lr: float = 0.001
    weight_decay: float = 0.01


@dataclass
class SGD(BaseOptimizer):
    _target_: str = "torch.optim.SGD"
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0005


config_store = ConfigStore.instance()
config_store.store(group="optimizer", name="adam", node=Adam)
config_store.store(group="optimizer", name="adamw", node=AdamW)
config_store.store(group="optimizer", name="sgd", node=SGD)
