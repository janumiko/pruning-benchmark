from config.main_config import MainConfig
from timm import create_model
import torch
from torch import nn


def register_models() -> None:
    """Register all the models in the timm registry by importing the module"""
    from architecture.models import lenet_cifar  # noqa: F401, I001
    from architecture.models import resnet_cifar  # noqa: F401


def construct_model(cfg: MainConfig) -> nn.Module:
    model: nn.Module = create_model(
        model_name=cfg.model,
        num_classes=cfg.dataset._num_classes,
    )
    model.load_state_dict(torch.load(cfg._checkpoint_path))
    return model
