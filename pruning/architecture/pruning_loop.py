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
from config.main_config import TYPES_TO_PRUNE, EarlyStopperConfig, Interval, MainConfig
from config.methods import BasePruningMethodConfig
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.utils.prune as prune
from wandb.sdk.wandb_run import Run

logger = logging.getLogger(__name__)


def start_pruning_experiment(cfg: MainConfig, out_directory: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    register_models()
    base_model: nn.Module = construct_model(cfg).to(device)
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
    base_top1acc = base_metrics["top1_accuracy"]
    base_top5acc = base_metrics["top5_accuracy"]
    logger.info(f"Base top-1 accuracy: {base_top1acc:.2f}%")
    logger.info(f"Base top-5 accuracy: {base_top5acc:.2f}%")

    metric_functions = {
        "top1_accuracy": utility.metrics.accuracy,
        "top5_accuracy": utility.metrics.top5_accuracy,
    }

    group_name = utility.summary.get_run_group_name(cfg, current_date)
    results_list = []

    for i in range(cfg._repeat):
        logger.info(f"Repeat {i+1}/{cfg._repeat}")

        wandb_run = utility.summary.create_wandb_run(
            cfg, group_name, f"repeat_{i+1}/{cfg._repeat}"
        )

        model = construct_model(cfg).to(device)
        optimizer = construct_optimizer(cfg, model)

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
            f"Total parameters to prune: {int(sum(pruning_steps) * total_params)} "
            f"({round(sum(pruning_steps) * 100, 2)}%)"
        )

        results = prune_model(
            model=model,
            pruning_config=cfg.pruning.method,
            early_stopper_config=cfg.early_stopper,
            optimizer=optimizer,
            loss_fn=cross_entropy,
            params_to_prune=params_to_prune,
            pruning_steps=pruning_steps,
            finetune_epochs=cfg.pruning.finetune_epochs,
            train_dl=train_dl,
            valid_dl=valid_dl,
            device=device,
            metrics_dict=metric_functions,
            wandb_run=wandb_run,
            checkpoints_interval=cfg.pruning._checkpoints_interval,
        )
        results["repeat"] = i + 1
        results_list.append(results)

        utility.pruning.log_parameters_sparsity(model, params_to_prune, logger)
        utility.pruning.log_module_sparsity(model, logger)

        if cfg._save_checkpoints:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(
                model.state_dict(),
                out_directory / f"{cfg.model}_{i}_{current_date}.pth",
            )

        wandb_run.summary["base_top1_accuracy"] = base_top1acc
        wandb_run.summary["base_top5_accuracy"] = base_top5acc

        wandb_run.finish()

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


def prune_model(
    model: nn.Module,
    pruning_config: BasePruningMethodConfig,
    early_stopper_config: EarlyStopperConfig,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    params_to_prune: Iterable[tuple[nn.Module, str]],
    pruning_steps: Iterable[int],
    finetune_epochs: int,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    metrics_dict: Mapping[str, Callable],
    wandb_run: Run,
    checkpoints_interval: Interval,
    device: torch.device,
) -> pd.DataFrame:
    """Prune the model using the given method.

    Args:
        model (nn.Module): The model to prune.
        pruning_config (BasePruningMethodConfig): The pruning config for the pruning method to use.
        early_stopper_config (EarlyStopperConfig): The early stopper to use for finetuning.
        optimizer (torch.optim.Optimizer): The optimizer to use for finetuning.
        loss_fn (nn.Module): The loss function to use for finetuning.
        params_to_prune (Iterable[tuple[nn.Module, str]]): The parameters to prune.
        pruning_steps (Iterable[int]): The number of parameters to prune at each step.
        finetune_epochs (int): The number of epochs to finetune the model.
        train_dl (torch.utils.data.DataLoader): The training dataloader.
        valid_dl (torch.utils.data.DataLoader): The validation dataloader.
        metrics_dict (Mapping[str, Callable]): The metrics to log during finetuning.
        wandb_run (Run): The wandb object to use for logging.
        checkpoints_interval (Interval): The interval to log checkpoints.
        device (torch.device): The device to use for training.
        early_stopper (None | utility.training.EarlyStopper, optional): The early stopper to use for finetuning. Defaults to None.

    Returns:
        pd.DataFrame: The metrics for the pruned checkpoints.
    """
    checkpoints_data = pd.DataFrame(
        columns=["pruned_precent", "top1_accuracy", "top5_accuracy", "total_epoch"]
    )
    total_epoch = 0

    early_stopper = utility.training.EarlyStopper(
        patience=early_stopper_config.patience,
        min_delta=early_stopper_config.min_delta,
        is_decreasing=early_stopper_config.metric.is_decreasing,
    )

    for iteration, step in enumerate(pruning_steps):
        logger.info(f"Pruning iteration {iteration + 1}/{len(pruning_steps)}")
        prune_module(params=params_to_prune, prune_percent=step, pruning_cfg=pruning_config)

        pruned, model_pruned = utility.pruning.calculate_pruning_ratio(model)
        iteration_info = {
            "iteration": iteration,
            "pruned_precent": round(pruned, 2),
            "model_pruned_precent": round(model_pruned, 2),
        }

        for epoch in range(finetune_epochs):
            logger.info(f"Epoch {epoch + 1}/{finetune_epochs}")
            total_epoch += 1

            train_loss = utility.training.train_epoch(
                module=model,
                train_dl=train_dl,
                loss_function=loss_fn,
                optimizer=optimizer,
                device=device,
            )

            metrics = utility.training.validate_epoch(
                module=model,
                valid_dl=valid_dl,
                loss_function=loss_fn,
                metrics_functions=metrics_dict,
                device=device,
            )

            for key, value in metrics.items():
                logger.info(f"{key}: {value:.4f}")

            # additonal epoch metrics
            metrics["training_loss"] = train_loss
            metrics["epoch"] = epoch + 1

            metrics.update(iteration_info)
            wandb_run.log(metrics)

            if early_stopper_config.enabled and early_stopper.check_stop(
                metrics[early_stopper_config.metric.name]
            ):
                logger.info(f"Early stopping after {epoch+1} epochs")
                early_stopper.reset()
                break

        if (
            checkpoints_interval.start * 100 <= pruned <= checkpoints_interval.end * 100
            and finetune_epochs > 0
        ):
            # post epoch metrics
            metrics["total_epoch"] = total_epoch

            checkpoints_data.loc[iteration] = {
                key: metrics[key] for key in checkpoints_data.columns
            }

    # summary info
    summary = wandb_run.summary
    summary["final_pruned_percent"] = round(pruned, 2)
    summary["total_epoch"] = total_epoch

    for module, name in params_to_prune:
        prune.remove(module, name)

    return checkpoints_data
