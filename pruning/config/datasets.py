from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


@dataclass
class BaseDataset:
    name: str = MISSING
    resize_value: Optional[int] = None
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


@dataclass
class ImageNet1K(BaseDataset):
    name: str = "imagenet1k"
    _path: str = "datasets/imagenet1k"
    _num_classes: int = 1000
