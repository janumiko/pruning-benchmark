from pathlib import Path

from architecture.construct_dataset import get_dataloaders
from architecture.construct_model import construct_model, register_models
from architecture.trainers.trainer_base import BaseTrainer
from architecture.utils import distributed_utils, training_utils
from architecture.utils.pylogger import RankedLogger
from config.main_config import MainConfig
import hydra
import torch
from torch import nn
import torch.distributed as dist
from torch_pruning.pruner.algorithms.metapruner import MetaPruner

logger = RankedLogger(__name__, rank_zero_only=True)


def start_pruning_experiment(cfg: MainConfig) -> None:
    """Start the pruning experiment.

    Args:
        cfg (MainConfig): Hydra config object with all the settings. (Located in config/main_config.py)
        hydra_output_dir (Path): The output directory for the Hydra experiment.
    """
    # set the seed
    training_utils.set_reproducibility(cfg.seed)
    metrcis_logger = ...

    register_models()
    rank = distributed_utils.get_rank()
    model = construct_model(cfg)

    train_loader, val_loader = get_dataloaders(cfg)
    scheluder = ...
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=f"cuda:{rank}",
        metrics_logger=metrcis_logger,
        distributed=cfg.distributed.enabled,
    )
    pruner = ...

    pruning_loop(cfg=cfg, model=model, scheduler=None, trainer=trainer, pruner=None)


def pruning_loop(
    cfg: MainConfig, model: nn.Module, scheduler, trainer: BaseTrainer, pruner: MetaPruner
):
    checkpoint_path = Path(f"{cfg.paths.output_dir}/model_checkpoint.pth")
    # validate the model before pruning
    trainer.validate(model)

    # for loop
    # prune the model
    # save the model
    save_model(model, checkpoint_path)

    # if distributed
    # load the model for each process
    if cfg.distributed.enabled:
        model = torch.load(checkpoint_path)
        dist.barrier()

    # create optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    # fit the model
    trainer.fit(model, optimizer)


@distributed_utils.rank_zero_only
def save_model(model: nn.Module, path: Path) -> None:
    torch.save(model, path)
