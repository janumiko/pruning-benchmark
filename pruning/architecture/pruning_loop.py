import datetime
import logging
from pathlib import Path
from typing import Callable, Iterable, Mapping

from architecture.construct_dataset import get_dataloaders
from architecture.construct_model import construct_model, register_models
from architecture.construct_optimizer import construct_optimizer
from architecture.pruning_methods.schedulers import construct_step_scheduler
import architecture.utility as utility
from config.main_config import Interval, MainConfig
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.utils.prune as prune
from wandb.sdk.wandb_run import Run

logger = logging.getLogger(__name__)
# TODO: somehow add the pruning classes to Hydra config
PRUNING_CLASSES = (nn.Linear, nn.Conv2d)


def start_pruning_experiment(cfg: MainConfig, out_directory: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    register_models()
    base_model: nn.Module = construct_model(cfg).to(device)
    train_dl, valid_dl = get_dataloaders(cfg)
    cross_entropy = nn.CrossEntropyLoss()

    early_stopper = None
    if cfg.early_stopper.enabled:
        early_stopper = utility.training.EarlyStopper(
            patience=cfg.early_stopper.patience,
            min_delta=cfg.early_stopper.min_delta,
        )

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

        params_to_prune = utility.pruning.get_parameters_to_prune(model, PRUNING_CLASSES)
        total_pruning_params = utility.pruning.calculate_parameters_amount(params_to_prune)
        pruning_steps = list(construct_step_scheduler(params_to_prune, cfg.pruning.scheduler))
        total_params = utility.pruning.get_parameter_count(model)

        logger.info(
            f"Iterations: {len(pruning_steps)}\n"
            f"Parameters to prune at each step: {pruning_steps}\n"
            f"Pruning percentages at each step {[round(step / total_pruning_params, 4) for step in pruning_steps]}\n"
            f"Total parameters to prune: {sum(pruning_steps)}/{total_params} "
            f"({round(sum(pruning_steps)/total_params, 4)*100}%)"
        )

        results = prune_model(
            model=model,
            method=prune.L1Unstructured,
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
            early_stopper=early_stopper,
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

    utility.summary.save_checkpoint_results(
        cfg, pd.concat(results_list), out_directory, group_name, base_top1acc, base_top5acc
    )


def prune_model(
    model: nn.Module,
    method: prune.BasePruningMethod,
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
    early_stopper: None | utility.training.EarlyStopper = None,
) -> pd.DataFrame:
    """Prune the model using the given method.

    Args:
        model (nn.Module): The model to prune.
        method (prune.BasePruningMethod): The method to use for pruning.
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
        columns=["pruned_precent", "top1_accuracy", "top5_accuracy", "epoch_mean", "epoch_std"]
    )
    epochs = []

    for iteration, step in enumerate(pruning_steps):
        logger.info(f"Pruning iteration {iteration + 1}/{len(pruning_steps)}")
        prune.global_unstructured(
            params_to_prune,
            pruning_method=method,
            amount=step,
        )

        pruned, model_pruned = utility.pruning.calculate_pruning_ratio(model)
        iteration_info = {
            "iteration": iteration,
            "pruned_precent": round(pruned, 2),
            "model_pruned_precent": round(model_pruned, 2),
        }

        for epoch in range(finetune_epochs):
            logger.info(f"Epoch {epoch + 1}/{finetune_epochs}")

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

            if early_stopper and early_stopper.check_stop(metrics["validation_loss"]):
                logger.info(f"Early stopping after {epoch+1} epochs")
                early_stopper.reset()
                epochs.append(epoch + 1)
                break

        if (
            checkpoints_interval.start * 100 <= pruned <= checkpoints_interval.end * 100
            and finetune_epochs > 0
        ):
            # post epoch metrics
            metrics["epoch_mean"] = np.mean(epochs) if epochs else finetune_epochs
            metrics["epoch_std"] = np.std(epochs) if epochs else 0

            checkpoints_data.loc[iteration] = {
                key: metrics[key] for key in checkpoints_data.columns
            }

    # summary info
    summary = wandb_run.summary
    summary["final_pruned_percent"] = round(pruned, 2)

    for module, name in params_to_prune:
        prune.remove(module, name)

    return checkpoints_data
