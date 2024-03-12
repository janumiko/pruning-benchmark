import torch
from architecture.models.lenet_cifar import LeNet
from architecture.models.resnet import ResNet18, ResNet50
from config.main_config import MainConfig
from torch import nn


def construct_lenet(path: str) -> nn.Module:
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


def construct_model(cfg: MainConfig) -> nn.Module:
    match (cfg.model.name.lower(), cfg.dataset.name.lower()):
        case ("lenet", _):
            return construct_lenet(cfg.model.checkpoint_path)
        case ("resnet50", _):
            return construct_resnet50(
                cfg.model.checkpoint_path, cfg.dataset.num_classes
            )
        case ("resnet18", _):
            return construct_resnet18(
                cfg.model.checkpoint_path, cfg.dataset.num_classes
            )
        case _:
            raise ValueError(
                f"Unknown model: {cfg.model.name} for dataset: {cfg.dataset.name}"
            )
