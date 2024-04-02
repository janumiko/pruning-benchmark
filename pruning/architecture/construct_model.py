from architecture.models.lenet_cifar import LeNet
from architecture.models.resnet import ResNet18, ResNet50
from config.main_config import MainConfig
import torch
from torch import nn
from torchvision.models import resnet18


def construct_lenet(path: str) -> nn.Module:
    model = LeNet()
    model.load_state_dict(torch.load(path))
    return model


def construct_resnet50_cifar(path: str, num_classes: int) -> nn.Module:
    model = ResNet50(num_classes)
    model.load_state_dict(torch.load(path))
    return model


def construct_resnet18_cifar(path: str, num_classes: int) -> nn.Module:
    model = ResNet18(num_classes)
    model.load_state_dict(torch.load(path))
    return model


def construct_resnet18_imagenet(path: str) -> nn.Module:
    model = resnet18()
    model.load_state_dict(torch.load(path))
    return model


def construct_model(cfg: MainConfig) -> nn.Module:
    match (cfg.model.name.lower(), cfg.dataset.name.lower()):
        case ("lenet", _):
            return construct_lenet(cfg.model._checkpoint_path)
        case ("resnet50", "cifar10" | "cifar100"):
            return construct_resnet50_cifar(cfg.model._checkpoint_path, cfg.dataset._num_classes)
        case ("resnet18", "cinfar10" | "cifar100"):
            return construct_resnet18_cifar(cfg.model._checkpoint_path, cfg.dataset._num_classes)
        case ("resnet18", "imagenet1k"):
            return construct_resnet18_imagenet(cfg.model._checkpoint_path)
        case _:
            raise ValueError(f"Unknown model: {cfg.model.name} for dataset: {cfg.dataset.name}")
