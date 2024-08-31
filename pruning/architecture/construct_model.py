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


def construct_model(cfg: MainConfig, rank: int) -> nn.Module:
    """
    Constructs a model based on the provided configuration and rank.

    Args:
        cfg (MainConfig): The main configuration object.
        rank (int): The rank of the model.

    Returns:
        nn.Module: The constructed model.
    """
    model: nn.Module = create_model(
        model_name=cfg.model,
    )
    model.to(rank)
    logger.info(
        model.load_state_dict(
            torch.load(cfg._checkpoint_path, map_location={"cuda:0": f"cuda:{rank}"})
        )
    )
    return DistributedDataParallel(model, device_ids=[rank])
