import datetime
import logging
from pathlib import Path
from typing import Callable

from architecture.construct_dataset import get_dataloaders
from architecture.construct_model import construct_model
from architecture.construct_optimizer import construct_optimizer
import architecture.utility as utility
from config.main_config import MainConfig
import torch
from torch import nn
import torch.nn.utils.prune as prune
import wandb
from wandb.sdk.wandb_run import Run

logger = logging.getLogger(__name__)


def start_pruning_experiment(cfg: MainConfig, out_directory: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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

    for i in range(cfg._repeat):
        logger.info(f"Repeat {i+1}/{cfg._repeat}")

        run_group_name = utility.summary.get_run_group_name(cfg, current_date)
        wandb_run = utility.summary.create_wandb_run(
            cfg, run_group_name, f"repeat_{i+1}/{cfg._repeat}"
        )

        model = construct_model(cfg).to(device)
        optimizer = construct_optimizer(cfg, model)
        pruning_parameters = utility.pruning.get_parameters_to_prune(
            model, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)
        )
        pruning_amount = int(
            round(
                utility.pruning.calculate_parameters_amount(pruning_parameters)
                * (cfg.pruning.step_percent / 100)
            )
        )
        total_count = utility.pruning.get_parameter_count(model)
        logger.info(
            f"Iterations: {cfg.pruning.iterations}\nPruning {pruning_amount} parameters per step\nTotal parameter count: {total_count}"
        )

        prune_model(
            model=model,
            method=prune.L1Unstructured,
            parameters_to_prune=pruning_parameters,
            optimizer=optimizer,
            loss_fn=cross_entropy,
            iterations=cfg.pruning.iterations,
            finetune_epochs=cfg.pruning.finetune_epochs,
            pruning_amount=pruning_amount,
            train_dl=train_dl,
            valid_dl=valid_dl,
            device=device,
            metrics_dict=metric_functions,
            wandb_run=wandb_run,
            pruning_checkpoints=cfg._wandb.pruning_checkpoints,
            early_stopper=early_stopper,
        )

        utility.pruning.log_parameters_sparsity(model, pruning_parameters, logger)
        utility.pruning.log_module_sparsity(model, logger)

        if cfg._save_checkpoints:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(
                model.state_dict(),
                out_directory / f"{cfg.model.name}_{i}_{current_date}.pth",
            )

        wandb_run.summary["base_top1_accuracy"] = base_top1acc
        wandb_run.summary["base_top5_accuracy"] = base_top5acc

        wandb_run.finish()


def prune_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    method: prune.BasePruningMethod,
    loss_fn: nn.Module,
    parameters_to_prune: list[tuple[nn.Module, str]],
    iterations: int,
    pruning_amount: int,
    finetune_epochs: int,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    metrics_dict: dict[str, Callable],
    wandb_run: Run,
    pruning_checkpoints: tuple[int],
    device: torch.device,
    early_stopper: None | utility.training.EarlyStopper = None,
) -> None:
    """Prune the model using the given method.

    Args:
        model (nn.Module): The model to prune.
        optimizer (torch.optim.Optimizer): The optimizer to use for finetuning.
        method (prune.BasePruningMethod): The method to use for pruning.
        loss_fn (nn.Module): The loss function to use for finetuning.
        parameters_to_prune (list[tuple[nn.Module, str]]): The parameters to prune.
        iterations (int): The amount of iterations to prune the model.
        pruning_amount (int): The amount of parameters to prune.
        finetune_epochs (int): The amount of epochs to finetune the model.
        train_dl (torch.utils.data.DataLoader): The training dataloader.
        valid_dl (torch.utils.data.DataLoader): The validation dataloader.
        metrics_dict (dict[str, Callable]): The metrics to log during finetuning.
        wandb_run (Run): The wandb object to use for logging.
        pruned_checkpoint (tuple[int]): The tuple of pruned checkpoints to save the metrics for.
        early_stopper (None | utility.training.EarlyStopper, optional): The early stopper to use for finetuning. Defaults to None.
        device (torch.device, optional): The device to use for training. Defaults to torch.device("cpu").
    """
    checkpoints_data = []
    checkpoints_keys = ["pruned_precent", "top1_accuracy", "top5_accuracy"]

    for iteration in range(iterations):
        logger.info(f"Pruning iteration {iteration + 1}/{iterations}")
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=method,
            amount=pruning_amount,
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

            metrics["training_loss"] = train_loss
            metrics["epoch"] = epoch + 1
            metrics.update(iteration_info)
            wandb_run.log(metrics)

            if early_stopper and early_stopper.check_stop(metrics["validation_loss"]):
                logger.info(f"Early stopping after {epoch+1} epochs")
                early_stopper.reset()
                break

        if round(pruned) in pruning_checkpoints:
            checkpoints_data.append([metrics[key] for key in checkpoints_keys])

    for module, name in parameters_to_prune:
        prune.remove(module, name)

    table = wandb.Table(data=checkpoints_data, columns=checkpoints_keys)
    wandb_run.log({"pruning_checkpoints": table})
