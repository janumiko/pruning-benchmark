from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class BaseDataset:
    name: str = MISSING
    _path: str = MISSING
    _num_classes: int = MISSING
    _download: bool = False


@dataclass
class CIFAR10(BaseDataset):
    name: str = "cifar10"
    _path: str = "datasets/cifar10"
    _num_classes: int = 10


@dataclass
class CIFAR100(BaseDataset):
    name: str = "cifar100"
    _path: str = "datasets/cifar100"
    _num_classes: int = 100
