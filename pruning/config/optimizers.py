from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class BaseOptimizer:
    name: str = MISSING
    lr: float = MISSING
    weight_decay: float = MISSING


@dataclass
class AdamW(BaseOptimizer):
    name: str = "adamw"
    lr: float = 0.001
    weight_decay: float = 0.01


@dataclass
class SGD(BaseOptimizer):
    name: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005
