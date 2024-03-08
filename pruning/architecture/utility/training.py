from pathlib import Path
import random
from typing import Callable
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_epoch(
    module: nn.Module,
    train_dl: DataLoader,
    optimizer: optim.Optimizer,
    loss_function: Callable,
    enable_autocast: bool = True,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Train a module for one epoch.

    Args:
        model (nn.Module): A PyTorch module.
        train_dl (DataLoader): Dataloader for the train data.
        optimizer (optim.Optimizer): Optimizer instance.
        loss_function (Callable): Loss function callable.
        enable_autocast (bool, optional): Whether to use automatic mixed precision. Defaults to True.
        device (torch.device, optional): Pytorch device. Defaults to torch.device("cpu").

    Returns:
        float: Average loss for the epoch.
    """

    module.train()
    train_loss = 0
    for inputs, labels in train_dl:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Compute prediction error
        with torch.cuda.amp.autocast(enabled=enable_autocast):
            prediction = module(inputs)
            loss = loss_function(prediction, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_dl)


def validate_epoch(
    module: nn.Module,
    valid_dl: DataLoader,
    loss_function: Callable,
    metrics_functions: dict[str, Callable],
    enable_autocast: bool = True,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """Validate the model on given data.

    Args:
        model (nn.Module): PyTorch module.
        valid_dl (DataLoader): Dataloader for the validation data.
        loss_function (Callable): Loss function callable.
        metrics_functions (dict[str, Callable]): Dictionary with metric_name : callable pairs.
        enable_autocast (bool, optional): Whether to use automatic mixed precision. Defaults to True.
        device (torch.device, optional): Pytorch device. Defaults to torch.device("cpu").

    Returns:
        dict[str, float]: Dictionary with average loss and metrics for the validation data.
    """

    module.eval()
    metrics = {name: 0.0 for name in metrics_functions.keys()}
    metrics["valid_loss"] = 0.0

    with torch.no_grad():
        for X, labels in valid_dl:
            X, labels = X.to(device), labels.to(device)

            # Compute prediction error
            with torch.cuda.amp.autocast(enabled=enable_autocast):
                pred = module(X)
                loss = loss_function(pred, labels)

            metrics["valid_loss"] += loss.item()

            for name, func in metrics_functions.items():
                metrics[name] += func(pred, labels)

    for name, _ in metrics.items():
        metrics[name] /= len(valid_dl)

    return metrics


def test(
    module: nn.Module,
    test_dl: DataLoader,
    loss_function: Callable,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """Test the model on the test dataset.

    Args:
        module (nn.Module): PyTorch module.
        test_dl (DataLoader): Dataloader for the test data.
        loss_function (Callable): Loss function callable.
        device (torch.device, optional): PyTorch device. Defaults to torch.device("cpu").

    Returns:
        tuple[float, float]: Average loss and accuracy for the test data.
    """

    size = len(test_dl.dataset)  # type: ignore
    num_batches = len(test_dl)
    module.eval()

    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in test_dl:
            X, y = X.to(device), y.to(device)
            pred = module(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = (correct / size) * 100

    return (test_loss, accuracy)


def accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate accuracy of the model.

    Args:
        output (torch.Tensor): Predicted output from the model.
        target (torch.Tensor): Correct labels for the data.

    Returns:
        float: The accuracy of the model in the range [0, 1].
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(labels)
        correct = 0
        correct += torch.sum(pred == labels).item()

    return correct / len(labels)


def set_reproducibility(seed: int) -> None:
    """Set the seed for reproducibility and deterministic behavior

    Args:
        seed (int): The seed to set for reproducibility
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_output_csv(path: Path, columns: list[str]) -> None:
    """Create a CSV file with the given columns

    Args:
        path (str): The path to the CSV file
        columns (list[str]): The columns for the CSV file
    """
    df = pd.DataFrame(columns=columns)
    df.to_csv(path, index=False, mode="w")
