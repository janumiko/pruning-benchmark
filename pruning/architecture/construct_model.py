import torch
from architecture.models.lenet_cifar import LeNet
from architecture.models.resnet import ResNet18, ResNet50
from omegaconf import DictConfig
from torch import nn


def construct_lenet(path: str, num_classes: int) -> nn.Module:
    model = LeNet()
    model.load_state_dict(torch.load(path))
    return model


def construct_resnet50(path: str, num_classes: int) -> nn.Module:
    model = ResNet50(num_classes)
    model.load_state_dict(torch.load(path))
    return model


def construct_resnet18(path: str, num_classes: int) -> nn.Module:
    model = ResNet18(num_classes)
    model.load_state_dict(torch.load(path))
    return model


def construct_model(cfg: DictConfig) -> nn.Module:
    match (cfg.model.name.lower(), cfg.dataset.name.lower()):
        case ("lenet", _):
            return construct_lenet(cfg.model.path, cfg.dataset.num_classes)
        case ("resnet50", _):
            return construct_resnet50(cfg.model.path, cfg.dataset.num_classes)
        case ("resnet18", _):
            return construct_resnet18(cfg.model.path, cfg.dataset.num_classes)
        case _:
            raise ValueError(
                f"Unknown model: {cfg.model.name} for dataset: {cfg.dataset.name}"
            )
