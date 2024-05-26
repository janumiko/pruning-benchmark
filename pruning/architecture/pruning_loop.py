import datetime
import logging
from pathlib import Path
from typing import Callable, Iterable, Mapping

from architecture.construct_dataset import get_dataloaders
from architecture.construct_model import construct_model, register_models
from architecture.construct_optimizer import construct_optimizer
from architecture.pruning_methods.methods import prune_module
from architecture.pruning_methods.schedulers import construct_step_scheduler
import architecture.utility as utility
from config.main_config import TYPES_TO_PRUNE, MainConfig
import pandas as pd
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.utils.prune as prune
from wandb.sdk.wandb_run import Run

logger = logging.getLogger(__name__)


def start_pruning_experiment(
    rank: int, world_size: int, cfg: MainConfig, out_directory: Path, ddp_init_method: str
) -> None:
    """Start the pruning experiment.

    Args:
        rank (int): The rank of the process.
        world_size (int): The number of processes.
        cfg (MainConfig): The configuration for the pruning experiment.
        out_directory (Path): The output directory for the experiment.
        ddp_init_method (str): The DDP initialization method.
    """
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    group_name = utility.summary.get_run_group_name(cfg, current_date)
    logger.info(f"Starting experiment at {current_date}")

    utility.training.setup_ddp(rank, world_size, ddp_init_method, cfg._seed)
    utility.summary.config_logger(out_directory, rank)
    device = torch.device(f"cuda:{rank}")

    register_models()
    base_model: nn.Module = construct_model(cfg, rank)
    train_dl, valid_dl = get_dataloaders(cfg)
    cross_entropy = nn.CrossEntropyLoss()

    base_metrics = utility.training.validate_epoch(
        module=base_model,
        valid_dl=valid_dl,
        loss_function=cross_entropy,
        metrics_functions={
            "top1_accuracy": utility.metrics.accuracy,
            "top5_accuracy": utility.metrics.top5_accuracy,
        },
        device=device,
    )
    base_metrics = utility.training.gather_metrics(base_metrics, world_size)
    base_top1acc = base_metrics["top1_accuracy"]
    base_top5acc = base_metrics["top5_accuracy"]
    logger.info(f"Base top-1 accuracy: {base_top1acc:.2f}%")
    logger.info(f"Base top-5 accuracy: {base_top5acc:.2f}%")

    metric_functions = {
        "top1_accuracy": utility.metrics.accuracy,
        "top5_accuracy": utility.metrics.top5_accuracy,
    }

    results_list = []

    if cfg.pruning.scheduler.name == "manual":
        utility.pruning.validate_manual_pruning(base_model, cfg, TYPES_TO_PRUNE)

    for i in range(cfg._repeat):
        logger.info(f"Repeat {i+1}/{cfg._repeat}")

        wandb_run = utility.summary.create_wandb_run(
            cfg,
            group_name,
            f"repeat_{i+1}/{cfg._repeat}",
            logging=(cfg._wandb.logging and rank == 0),
        )

        model = construct_model(cfg, rank)

        if (
            "structured" in cfg.pruning.method.name
            and "unstructured" not in cfg.pruning.method.name
        ):
            # add batchnorm layer to pruned parameters in case of structured
            # needed to remove the corresponding batchnorm channels when pruning layers
            params_to_prune = utility.pruning.get_parameters_to_prune(
                model, (*TYPES_TO_PRUNE, nn.BatchNorm2d)
            )
        else:
            params_to_prune = utility.pruning.get_parameters_to_prune(model, TYPES_TO_PRUNE)

        pruning_steps = list(construct_step_scheduler(cfg.pruning.scheduler))
        total_params = utility.pruning.get_parameter_count(model)

        logger.info(
            f"Iterations: {len(pruning_steps)}\n"
            f"Pruning percentages at each step {pruning_steps}\n"
            f"Total parameters to prune: {int(sum([sum(step) for step in pruning_steps]) * total_params)} "
            f"({round(sum([sum(step) for step in pruning_steps]) * 100, 2)}%)"
        )

        results = prune_model(
            rank=rank,
            world_size=world_size,
            model=model,
            cfg=cfg,
            out_directory=out_directory,
            loss_fn=cross_entropy,
            params_to_prune=params_to_prune,
            pruning_steps=pruning_steps,
            train_dl=train_dl,
            valid_dl=valid_dl,
            device=device,
            metrics_dict=metric_functions,
            wandb_run=wandb_run,
        )
        results["repeat"] = i + 1
        results_list.append(results)

        utility.pruning.log_parameters_sparsity(model, params_to_prune, logger)
        utility.pruning.log_module_sparsity(model, logger)

        if cfg._save_checkpoints and rank == 0:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(
                model.state_dict(),
                out_directory / f"{cfg.model}_{i}_{current_date}.pth",
            )

        wandb_run.summary["base_top1_accuracy"] = base_top1acc
        wandb_run.summary["base_top5_accuracy"] = base_top5acc

        wandb_run.finish()

    if rank == 0:
        iterations = len(list(construct_step_scheduler(cfg.pruning.scheduler)))
        utility.summary.save_checkpoint_results(
            cfg,
            pd.concat(results_list),
            out_directory,
            group_name,
            iterations,
            base_top1acc,
            base_top5acc,
        )

    utility.training.cleanup_ddp()


def prune_model(
    rank: int,
    world_size: int,
    cfg: MainConfig,
    out_directory: Path,
    model: nn.Module,
    loss_fn: nn.Module,
    params_to_prune: Iterable[tuple[nn.Module, str]],
    pruning_steps: Iterable[int],
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    metrics_dict: Mapping[str, Callable],
    wandb_run: Run,
    device: torch.device,
) -> pd.DataFrame:
    """Prune the model using the given method.

    Args:
        rank (int): The rank of the process.
        world_size (int): The number of processes.
        cfg (MainConfig): The configuration for the pruning method.
        out_directory (Path): The output directory for the experiment.
        model (nn.Module): The model to prune.
        loss_fn (nn.Module): The loss function to use for finetuning.
        params_to_prune (Iterable[tuple[nn.Module, str]]): The parameters to prune.
        pruning_steps (Iterable[int]): The number of parameters to prune at each step.
        train_dl (torch.utils.data.DataLoader): The training dataloader.
        valid_dl (torch.utils.data.DataLoader): The validation dataloader.
        metrics_dict (Mapping[str, Callable]): The metrics to log during finetuning.
        wandb_run (Run): The wandb object to use for logging.
        device (torch.device): The device to use for training.

    Returns:
        pd.DataFrame: The metrics for the pruned checkpoints.
    """
    checkpoints_data = pd.DataFrame(
        columns=["pruned_precent", "top1_accuracy", "top5_accuracy", "total_epoch"]
    )
    total_epoch = 0
    pruned_checker = 0

    early_stopper = utility.training.EarlyStopper(
        patience=cfg.early_stopper.patience,
        min_delta=cfg.early_stopper.min_delta,
        is_decreasing=cfg.early_stopper.metric.is_decreasing,
    )

    checkpoint_criterion = cfg.best_checkpoint_criterion
    checkpoint_path = Path(f"{out_directory}/best_checkpoint.pth")
    best_checkpoint = {
        "state_dict": None,
        checkpoint_criterion.name: float("inf" if checkpoint_criterion.is_decreasing else "-inf"),
        "epoch": None,
        "metrics": None,
    }

    for iteration, pruning_values in enumerate(pruning_steps):
        logger.info(f"Pruning iteration {iteration + 1}/{len(pruning_steps)}")
        pruned_checker += sum(pruning_values)

        # load last best checkpoint state dict
        if iteration and checkpoint_path.exists():
            model.load_state_dict(
                torch.load(checkpoint_path, map_location={"cuda:0": f"cuda:{rank}"})
            )

        if iteration == 0 or rank == 0:
            prune_module(
                params=params_to_prune,
                pruning_values=pruning_values,
                pruning_cfg=cfg.pruning.method,
            )

        logger.debug("Broadcasting buffers")
        for name, buffer in model.named_buffers():
            if name.endswith("_mask"):
                dist.broadcast(buffer, src=0, async_op=True)
        dist.barrier()

        # reset optimizer in each pruning iteration
        logger.debug("Constructing optimizer")
        optimizer = construct_optimizer(cfg, model)

        logger.debug("Creating checkpoint")
        best_checkpoint["state_dict"] = model.state_dict()
        best_checkpoint["metrics"] = {}
        best_checkpoint["epoch"] = 0
        best_checkpoint[checkpoint_criterion.name] = float(
            "inf" if checkpoint_criterion.is_decreasing else "-inf"
        )

        pruned, model_pruned = utility.pruning.calculate_pruning_ratio(model)
        logger.info(f"Pruned: {pruned:.2f}%")
        logger.info(f"Model pruned: {model_pruned:.2f}%")

        if abs(pruned - pruned_checker) > 0.01:
            logger.warning(
                f"Pruned and pruned_checker percentages do not match: {round(pruned, 2)} != {round(pruned_checker, 2)}"
            )

        iteration_info = {
            "iteration": iteration,
            "pruned_precent": round(pruned, 2),
            "model_pruned_precent": round(model_pruned, 2),
        }

        for epoch in range(cfg.pruning.finetune_epochs):
            logger.info(f"Epoch {epoch + 1}/{cfg.pruning.finetune_epochs}")
            total_epoch += 1

            logger.debug("Training epoch")
            train_loss = utility.training.train_epoch(
                module=model,
                train_dl=train_dl,
                loss_function=loss_fn,
                optimizer=optimizer,
                device=device,
            )

            logger.debug("Validating epoch")
            metrics = utility.training.validate_epoch(
                module=model,
                valid_dl=valid_dl,
                loss_function=loss_fn,
                metrics_functions=metrics_dict,
                device=device,
            )
            metrics["training_loss"] = train_loss
            metrics = utility.training.gather_metrics(metrics, world_size)

            for key, value in metrics.items():
                logger.info(f"{key}: {value:.4f}")

            # additonal epoch metrics
            metrics["epoch"] = epoch + 1
            metrics.update(iteration_info)
            wandb_run.log(metrics)

            if checkpoint_criterion.is_decreasing == (
                metrics[checkpoint_criterion.name] < best_checkpoint[checkpoint_criterion.name]
            ):
                logger.info(
                    f"New best checkpoint found: {checkpoint_criterion.name}: {metrics[checkpoint_criterion.name]:.4f}"
                )
                best_checkpoint["state_dict"] = model.state_dict()
                best_checkpoint[checkpoint_criterion.name] = metrics[checkpoint_criterion.name]
                best_checkpoint["epoch"] = epoch + 1
                best_checkpoint["metrics"] = metrics

            if cfg.early_stopper.enabled and early_stopper.check_stop(
                metrics[cfg.early_stopper.metric.name]
            ):
                logger.info(f"Early stopping after {epoch+1} epochs")
                early_stopper.reset()
                break

        logger.info(f"Best checkpoint saved on epoch: {best_checkpoint['epoch']}")

        if rank == 0:
            logger.debug(f"Saving best checkpoint to {checkpoint_path}")
            torch.save(best_checkpoint["state_dict"], checkpoint_path)
        dist.barrier()

        if (
            cfg.pruning._checkpoints_interval.start * 100
            <= pruned
            <= cfg.pruning._checkpoints_interval.end * 100
            and cfg.pruning.finetune_epochs > 0
        ):
            # post epoch metrics
            metrics["total_epoch"] = total_epoch
            best_checkpoint["metrics"]["total_epoch"] = total_epoch

            checkpoints_data.loc[iteration] = {
                key: best_checkpoint["metrics"][key] for key in checkpoints_data.columns
            }

    if rank == 0:
        if checkpoint_path.exists():
            logging.debug(f"Removing checkpoint file {checkpoint_path}")
            checkpoint_path.unlink()

    # summary info
    summary = wandb_run.summary
    summary["final_pruned_percent"] = round(pruned, 2)
    summary["total_epoch"] = total_epoch

    for module, name in params_to_prune:
        prune.remove(module, name)

    return checkpoints_data
