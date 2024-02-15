from typing import Any
import torch
import torch.nn.utils.prune as prune
from torch import nn
import architecture.utility as utility


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
    device: torch.device = torch.device("cpu"),
    wandb_run: None | Any = None,
) -> nn.Module:
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
        device (torch.device, optional): The device to use for training. Defaults to torch.device("cpu").
        wandb_run (None | Any, optional): The wandb object to use for logging. Defaults to None.

    Returns:
        nn.Module: Pruned model.
    """

    for iteration in range(iterations):
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=method,
            amount=pruning_amount,
        )

        print(f"Pruning iteration {iteration + 1}/{iterations}")
        for epoch in range(finetune_epochs):
            utility.training.train_epoch(
                module=model,
                train_dl=train_dl,
                loss_function=loss_fn,
                optimizer=optimizer,
                device=device,
            )

            valid_loss, valid_accuracy = utility.training.validate(
                module=model,
                valid_dl=valid_dl,
                loss_function=loss_fn,
                device=device,
            )

            print(
                f"Epoch {epoch + 1}/{finetune_epochs} - "
                f"Validation loss: {valid_loss:.4f}, "
                f"Validation accuracy: {valid_accuracy:.2f}%"
            )

            if wandb_run:
                wandb_run.log(
                    {
                        "valid_loss": valid_loss,
                        "valid_accuracy": valid_accuracy,
                    }
                )

    for module, name in parameters_to_prune:
        prune.remove(module, name)

    sparsity = 1.0 - utility.pruning.calculate_total_sparsity(
        model, parameters_to_prune
    )
    print(f"Final sparsity: {sparsity:.2f}")

    return model
