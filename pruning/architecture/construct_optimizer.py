import torch
from config.main_config import MainConfig
from torch import nn


def construct_optimizer(cfg: MainConfig, model: nn.Module) -> torch.optim.Optimizer:
    """Construct optimizer for the given model.

    Args:
        cfg (MainConfig): Config dictionary.
        model (nn.Module): PyTorch model.

    Returns:
        torch.optim.Optimizer: PyTorch optimizer.
    """

    match cfg.optimizer.name.lower():
        case "adamw":
            return torch.optim.Adam(
                model.parameters(),
                lr=cfg.optimizer.learning_rate,
                weight_decay=cfg.optimizer.weight_decay,
            )
        case "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=cfg.optimizer.learning_rate,
                momentum=cfg.optimizer.momentum,
                weight_decay=cfg.optimizer.weight_decay,
            )
        case _:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer.name}")
