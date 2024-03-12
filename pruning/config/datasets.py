from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class BaseDataset:
    name: str = MISSING
    path: str = MISSING
    num_of_classes: int = MISSING
    download: bool = False


@dataclass
class CIFAR10(BaseDataset):
    name: str = "cifar10"
    path: str = "datasets/cifar10"
    num_of_classes: int = 10


@dataclass
class CIFAR100(BaseDataset):
    name: str = "cifar100"
    path: str = "datasets/cifar100"
    num_of_classes: int = 100
