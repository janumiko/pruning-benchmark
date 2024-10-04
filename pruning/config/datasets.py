from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BaseDataset:
    name: str = MISSING
    resize_value: Optional[int] = None
    crop_value: Optional[int] = None
    num_classes: int = MISSING
    path: str = MISSING
    download: bool = False


@dataclass
class CIFAR10(BaseDataset):
    name: str = "cifar10"
    num_classes: int = 10
    path: str = "datasets/cifar10"


@dataclass
class CIFAR100(BaseDataset):
    name: str = "cifar100"
    path: str = "datasets/cifar100"
    num_classes: int = 100


@dataclass
class ImageNet1K(BaseDataset):
    name: str = "imagenet1k"
    path: str = "datasets/imagenet1k"
    num_classes: int = 1000


config_store = ConfigStore.instance()
config_store.store(group="dataset", name="cifar10", node=CIFAR10)
config_store.store(group="dataset", name="cifar100", node=CIFAR100)
config_store.store(group="dataset", name="imagenet1k", node=ImageNet1K)