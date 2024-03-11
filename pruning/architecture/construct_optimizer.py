import torch
from omegaconf import DictConfig
from torch import nn


def construct_optimizer(cfg: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    """Construct optimizer for the given model.

    Args:
        cfg (DictConfig): Config dictionary.
        model (nn.Module): PyTorch model.

    Returns:
        torch.optim.Optimizer: PyTorch optimizer.
    """

    match cfg.optimizer.name.lower():
        case "adamw":
            return torch.optim.Adam(
                model.parameters(),
                lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
            )
        case "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=cfg.optimizer.lr,
                momentum=cfg.optimizer.momentum,
                weight_decay=cfg.optimizer.weight_decay,
            )
        case _:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer.name}")
