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
from architecture.utils.metrics import BaseMetricLogger
from omegaconf import OmegaConf


logger = RankedLogger(__name__, rank_zero_only=True)


def start_pruning_experiment(cfg: MainConfig) -> None:
    """Start the pruning experiment.

    Args:
        cfg (MainConfig): Hydra config object with all the settings. (Located in config/main_config.py)
        hydra_output_dir (Path): The output directory for the Hydra experiment.
    """
    # set the seed
    training_utils.set_reproducibility(cfg.seed)

    register_models()
    rank = distributed_utils.get_rank()
    model = construct_model(cfg)
    model = model.to(f"cuda:{rank}")

    train_loader, val_loader = get_dataloaders(cfg)
    example_inputs: torch.Tensor = next(iter(train_loader))[0]
    example_inputs = example_inputs.to(f"cuda:{rank}")

    metrcis_logger: BaseMetricLogger = hydra.utils.instantiate(
        cfg.wandb,
        main_config=OmegaConf.to_container(cfg),
        log_path=cfg.paths.output_dir,
        _recursive_=False,
    )
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        dataset_config=cfg.dataset,
        device=f"cuda:{rank}",
        metrics_logger=metrcis_logger,
        distributed=cfg.distributed.enabled,
    )
    pruner = hydra.utils.instantiate(
        cfg.pruner,
        model=model,
        example_inputs=example_inputs,
    )

    pruning_loop(
        cfg=cfg, model=model, trainer=trainer, pruner=pruner, metrics_logger=metrcis_logger
    )


def pruning_loop(
    cfg: MainConfig,
    model: nn.Module,
    trainer: BaseTrainer,
    pruner: BasePruner,
    metrics_logger: BaseMetricLogger,
) -> None:
    checkpoint_path = Path(f"{cfg.paths.output_dir}/model_checkpoint.pth")
    base_metrics = trainer.validate(model)
    base_metrics = {f"base_{k}": v for k, v in base_metrics.items()}
    base_metrics["base_nparams"] = sum(p.numel() for p in model.parameters())
    metrics_logger.summary(base_metrics)

    for step in range(pruner.steps):
        logger.info(f"Pruning step {step+1}/{pruner.steps}")

        prune_model(pruner, checkpoint_path)
        metrics_logger.log(pruner.statistics(), commit=False)

        if cfg.distributed.enabled:
            model = torch.load(checkpoint_path)
            dist.barrier()

        optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
        trainer.fit(model, optimizer)

    final_metrics = trainer.validate(model)
    final_metrics["total_epochs"] = trainer.total_epochs
    logger.info(f"Final metrics: {final_metrics}")
    metrics_logger.summary(final_metrics)


@distributed_utils.rank_zero_only
def prune_model(pruner: BasePruner, checkpoint_path: str) -> None:
    pruner.step()
    logger.info(pruner.statistics())
    save_model(pruner.model, checkpoint_path)


@distributed_utils.rank_zero_only
def save_model(model: nn.Module, path: Path) -> None:
    logger.info(f"Saving model to {path}")
    torch.save(model, path)
