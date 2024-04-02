from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class BaseModel:
    name: str = MISSING
    _checkpoint_path: str = MISSING


@dataclass
class ResNet18_CIFAR10(BaseModel):
    name: str = "resnet18"
    _checkpoint_path: str = "checkpoints/resnet18_cifar10.pth"


@dataclass
class ResNet18_CIFAR100(BaseModel):
    name: str = "resnet18"
    _checkpoint_path: str = "checkpoints/resnet18_cifar100.pth"


@dataclass
class ResNet18_ImageNet1K(BaseModel):
    name: str = "resnet18"
    _checkpoint_path: str = "checkpoints/resnet18_imagenet1k.pth"


@dataclass
class LeNet_CIFAR10(BaseModel):
    name: str = "lenet"
    _checkpoint_path: str = "checkpoints/lenet_cifar10.pth"
