import torch

from architecture.models.lenet import LeNet
from omegaconf import DictConfig
from torch import nn


def construct_lenet(path: str) -> nn.Module:
    model = LeNet()
    model.load_state_dict(torch.load(path))
    return model


def construct_model(cfg: DictConfig) -> nn.Module:
    match (cfg.model.name.lower(), cfg.dataset.name.lower()):
        case ("lenet", "cifar10"):
            return construct_lenet(cfg.model.path)
        case _:
            raise ValueError(
                f"Unknown model: {cfg.model.name} for dataset: {cfg.dataset.name}"
            )
