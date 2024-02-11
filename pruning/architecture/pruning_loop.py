import torch
import torch.nn.utils.prune as prune
from torch import nn
import architecture.utility as utility
import wandb


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
    logging: bool = False,
) -> nn.Module:
    """Prune the model using the given method.

    Args:
        model (nn.Module): Model to prune.
        method (prune.BasePruningMethod): Pruning method to use.
        parameters_to_prune (list[tuple[nn.Module, str]]): List of parameters to prune.
        iterations (int): Number of iterations to prune.
        pruning_amount (int): Amount to prune.
        finetune_epochs (int): Number of epochs to finetune the model.
        train_dl (torch.utils.data.DataLoader): Training dataloader.
        valid_dl (torch.utils.data.DataLoader): Validation dataloader.
        device (torch.device, optional): Device to use. Defaults to torch.device("cpu").

    Returns:
        nn.Module: Pruned model.
    """

    if logging:
        # Initialize a new wandb run
        wandb.init(project="pruning", name="prune_model")

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

            if logging:
                wandb.log(
                    {
                        "valid_loss": valid_loss,
                        "valid_accuracy": valid_accuracy,
                    }
                )

    for module, name in parameters_to_prune:
        prune.remove(module, name)

    return model
