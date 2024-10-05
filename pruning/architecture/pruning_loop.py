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
from architecture.pruners.pruner_base import BasePruner

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
    model = model.to(f"cuda:{rank}")

    train_loader, val_loader = get_dataloaders(cfg)
    example_inputs = next(iter(train_loader))[0]
    example_inputs = example_inputs.to(f"cuda:{rank}")

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=f"cuda:{rank}",
        metrics_logger=metrcis_logger,
        distributed=cfg.distributed.enabled,
    )
    pruner = hydra.utils.instantiate(
        cfg.pruner,
        model=model,
        example_inputs=example_inputs,
    )

    pruning_loop(cfg=cfg, rank=rank, model=model, trainer=trainer, pruner=pruner)


def pruning_loop(
    cfg: MainConfig, rank: int, model: nn.Module, trainer: BaseTrainer, pruner: BasePruner
):
    checkpoint_path = Path(f"{cfg.paths.output_dir}/model_checkpoint.pth")
    # validate the model before pruning
    trainer.validate(model)

    for step in range(pruner.scheduler_steps):
        logger.info(f"Pruning step {step+1}/{pruner.scheduler_steps}")
        # for loop
        # prune the model
        # save the model
        prune_model(pruner, checkpoint_path)

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
def prune_model(pruner: BasePruner, checkpoint_path: str) -> None:
    print(pruner.statistics())
    print(pruner._pruner.current_step)
    pruner.step()
    print(pruner.statistics())
    save_model(pruner.model, checkpoint_path)



@distributed_utils.rank_zero_only
def save_model(model: nn.Module, path: Path) -> None:
    logger.info(f"Saving model to {path}")
    torch.save(model, path)
