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
    metrics_functions: dict[str, Callable],
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


def accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate accuracy of the model. An alias for topk_accuracy with k=1.

    Args:
        output (torch.Tensor): Predicted output from the model.
        target (torch.Tensor): Correct labels for the data.

    Returns:
        float: The accuracy of the model in the range [0, 1].
    """
    return topk_accuracy(output, labels, topk=1)


def top5_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate top-5 accuracy of the model. An alias for topk_accuracy with k=5.

    Args:
        prediction (torch.Tensor): Predicted output from the model.
        target (torch.Tensor): Correct labels for the data.

    Returns:
        float: The top-5 accuracy of the model in the range [0, 1].
    """
    return topk_accuracy(prediction, target, topk=5)


def topk_accuracy(prediction: torch.Tensor, target: torch.Tensor, topk) -> float:
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        prediction (torch.Tensor): Prediction tensor with shape (batch_size, num_classes).
        target (torch.Tensor: Ground truth tensor with shape (batch_size).
        topk (int): The values of k to compute the accuracy over.

    Returns:
        float: The top_k accuracy of the model.
    """

    with torch.no_grad():
        top5_pred = torch.topk(prediction, k=topk, dim=1).indices
        assert top5_pred.shape[0] == len(target)
        correct = 0
        for i in range(topk):
            correct += torch.sum(top5_pred[:, i] == target).item()

    return (correct * 100) / len(target)


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
