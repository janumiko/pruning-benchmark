from pathlib import Path
import random
from typing import Callable
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .metrics import accuracy, top5_accuracy
from .loggers import BaseMetricLogger


def train_epoch(
    module: nn.Module,
    train_dl: DataLoader,
    optimizer: optim.Optimizer,
    loss_function: Callable,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Train a module for one epoch.

    Args:
        model (nn.Module): A PyTorch module.
        train_dl (DataLoader): Dataloader for the train data.
        optimizer (optim.Optimizer): Optimizer instance.
        loss_function (Callable): Loss function callable.
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
    loggers: list[BaseMetricLogger],
    device: torch.device = torch.device("cpu"),
) -> float:
    """Validate the model on given data.

    Args:
        module (nn.Module): PyTorch module.
        valid_dl (DataLoader): Dataloader for the validation data.
        loss_function (Callable): Loss function callable.
        loggers (list[BaseMetricLogger]): List of metric loggers.
        device (torch.device, optional): PyTorch device. Defaults to torch.device("cpu").

    Returns:
        float: Average loss for the validation data.
    """

    module.eval()

    with torch.no_grad():
        for X, labels in valid_dl:
            X, labels = X.to(device), labels.to(device)

            # Compute prediction error
            pred = module(X)
            loss = loss_function(pred, labels)

            for logger in loggers:
                logger.log_batch(pred, labels)

    for logger in loggers:
        logger.log_epoch()

    return loss


def test(
    module: nn.Module,
    test_dl: DataLoader,
    loss_function: Callable,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float, float]:
    """Test the model on the test dataset.

    Args:
        module (nn.Module): PyTorch module.
        test_dl (DataLoader): Dataloader for the test data.
        loss_function (Callable): Loss function callable.
        device (torch.device, optional): PyTorch device. Defaults to torch.device("cpu").

    Returns:
        tuple[float, float]: Average loss and accuracy for the test data.
    """

    num_batches = len(test_dl)
    module.eval()

    test_loss, acc, top5_acc = 0.0, 0.0, 0.0
    with torch.no_grad():
        for X, y in test_dl:
            X, y = X.to(device), y.to(device)
            pred = module(X)
            test_loss += loss_function(pred, y).item()
            acc += accuracy(pred, y)
            top5_acc += top5_accuracy(pred, y)

    test_loss /= num_batches
    acc /= num_batches
    top5_acc /= num_batches

    return (test_loss, acc, top5_acc)


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


class EarlyStopper:
    """Early stopping class.
    Monitors validation loss and stops training if it does not improve after a given patience.
    Improvement is defined as validation loss < min_validation_loss + min_delta.
    Patience is the number of epochs to wait for improvement before stopping.
    """

    def __init__(self, patience: int, min_delta: int) -> None:
        """Initialize EarlyStopper.

        Args:
            patience (int): Number of epochs to wait for improvement.
            min_delta (int): Delta which which is used to decide epoch as improvement.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def check_stop(self, validation_loss: float) -> bool:
        """Check if training should be stopped.

        Args:
            validation_loss (float): Validation loss.

        Returns:
            bool: Boolean indicating whether training should be stopped.
        """

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1

        return self.counter >= self.patience
