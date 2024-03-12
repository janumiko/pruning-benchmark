from typing import Any
import torch
import torch.nn.utils.prune as prune
from torch import nn
import architecture.utility as utility
import logging

logger = logging.getLogger(__name__)


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
    early_stopper: None | utility.training.EarlyStopper = None,
    device: torch.device = torch.device("cpu"),
    wandb_run: None | Any = None,
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
        early_stopper (None | utility.training.EarlyStopper, optional): The early stopper to use for finetuning. Defaults to None.
        device (torch.device, optional): The device to use for training. Defaults to torch.device("cpu").
        wandb_run (None | Any, optional): The wandb object to use for logging. Defaults to None.
    """

    total_count = utility.pruning.get_parameter_count(model)
    logger.info(
        f"Iterations: {iterations}\nPruning {pruning_amount} parameters per step\nTotal parameter count: {total_count}"
    )

    top1_logger = utility.loggers.AccuracyLogger(
        "top-1 accuracy", logger, is_epoch_logging=True
    )
    top5_logger = utility.loggers.AccuracyLogger(
        "top-5 accuracy", logger, topk=5, is_epoch_logging=True
    )
    metric_loggers = [top1_logger, top5_logger]

    for iteration in range(iterations):
        logger.info(f"Pruning iteration {iteration + 1}/{iterations}")
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=method,
            amount=pruning_amount,
        )

        for epoch in range(finetune_epochs):
            logger.info(f"Epoch {epoch + 1}/{finetune_epochs}")

            utility.training.train_epoch(
                module=model,
                train_dl=train_dl,
                loss_function=loss_fn,
                optimizer=optimizer,
                device=device,
            )

            valid_loss = utility.training.validate_epoch(
                module=model,
                valid_dl=valid_dl,
                loss_function=loss_fn,
                loggers=metric_loggers,
                device=device,
            )

            logger.info(f"Validation loss: {valid_loss:.4f}")

            if wandb_run:
                wandb_run.log({"validation_loss": valid_loss})
                for metric in metric_loggers:
                    wandb_run.log({metric.metric_name: metric.epoch_history[-1]})

            if early_stopper and early_stopper.check_stop(valid_loss):
                logger.info(f"Early stopping after {epoch+1} epochs")
                early_stopper.reset()
                break

    for module, name in parameters_to_prune:
        prune.remove(module, name)

    pruned_sparsity = 100 - utility.pruning.calculate_parameters_sparsity(
        model, parameters_to_prune
    )
    total_sparsity = 100 - utility.pruning.calculate_total_sparsity(model)
    logger.info(
        f"Sparsity: {pruned_sparsity:.2f}% of non-zero values for pruned parameters."
    )
    logger.info(f"Total sparsity of the model {total_sparsity:.2f}%")
