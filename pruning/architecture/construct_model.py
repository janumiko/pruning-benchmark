import logging

from config.main_config import MainConfig
from timm import create_model
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger(__name__)


def register_models() -> None:
    """Register all the models in the timm registry by importing the module"""
    import architecture.models  # noqa: F401


def construct_model(cfg: MainConfig) -> nn.Module:
    model: nn.Module = create_model(
        model_name=cfg.model.name,
        num_classes=cfg.dataset.num_classes,
    )
    logger.info(model.load_state_dict(torch.load(cfg.model.checkpoint_path, map_location="cpu")))

    return model
