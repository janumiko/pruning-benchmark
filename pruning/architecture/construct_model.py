import torch
from architecture.lenet import LeNet
from omegaconf import DictConfig
from torch import nn
from torchvision.models import resnet50, resnet18


def construct_lenet(path: str) -> nn.Module:
    model = LeNet()
    model.load_state_dict(torch.load(path))
    return model


def construct_resnet50(path: str) -> nn.Module:
    model = resnet50()
    model.load_state_dict(torch.load(path))
    return model


def construct_resnet18(path: str) -> nn.Module:
    model = resnet18()
    model.load_state_dict(torch.load(path))
    return model


def construct_model(cfg: DictConfig) -> nn.Module:
    match (cfg.model.name.lower(), cfg.dataset.name.lower()):
        case ("lenet", "cifar10"):
            return construct_lenet(cfg.model.path)
        case ("resnet50", "cifar10"):
            return construct_resnet50(cfg.model.path)
        case ("resnet18", "cifar100"):
            return construct_resnet18(cfg.model.path)
        case _:
            raise ValueError(
                f"Unknown model: {cfg.model.name} for dataset: {cfg.dataset.name}"
            )
