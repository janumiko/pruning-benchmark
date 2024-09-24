from pathlib import Path

from architecture.utils.pylogger import RankedLogger
from config.main_config import MainConfig
from torch import nn
from architecture.trainers.trainer_base import Trainer
from torch_pruning.pruner.algorithms.metapruner import MetaPruner
import torch


logger = RankedLogger(__name__, rank_zero_only=True)


def start_pruning_experiment(cfg: MainConfig) -> None:
    """Start the pruning experiment.

    Args:
        cfg (MainConfig): Hydra config object with all the settings. (Located in config/main_config.py)
        hydra_output_dir (Path): The output directory for the Hydra experiment.
    """
    # set the seed

    metrcis_logger = ...
    model = ...

    train_loader, val_loader = ...
    scheluder = ...
    trainer = ...
    pruner = ...

    # pruning_loop()


def pruning_loop(
    cfg: MainConfig, model: nn.Module, scheduler, trainer: Trainer, pruner: MetaPruner
):
    pass
    # validate the model before pruning
    trainer.validate(model)

    # for loop
    # prune the model
    # save the model

    # if distributed
    # load the model for each process
    if cfg.distributed.enabled:
        model = torch.load("path_to_model.pth")

    # create optimizer
    optimizer = ...
    # fit the model
    trainer.fit(model, optimizer)
