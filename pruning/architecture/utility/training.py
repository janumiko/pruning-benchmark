from typing import Callable

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

        # Compute prediction error
        prediction = module(inputs)
        loss = loss_function(prediction, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    return train_loss / len(train_dl)


def validate(
    module: nn.Module,
    valid_dl: DataLoader,
    loss_function: Callable,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """Validate the model on given data.

    Args:
        model (nn.Module): PyTorch module.
        valid_dl (DataLoader): Dataloader for the validation data.
        loss_function (Callable): Loss function callable.
        device (torch.device, optional): Pytorch device. Defaults to torch.device("cpu").

    Returns:
        tuple[float, float]: Average loss and accuracy for the validation data.
    """

    module.eval()
    valid_loss = 0.0
    valid_accuracy = 0.0
    with torch.no_grad():
        for X, y in valid_dl:
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = module(X)
            loss = loss_function(pred, y)

            # Compute accuracy
            valid_accuracy += (pred.argmax(1) == y).float().mean()
            valid_loss += loss.item()

    valid_loss /= len(valid_dl)
    valid_accuracy /= len(valid_dl)

    return (valid_loss, valid_accuracy)


def test(
    model: nn.Module,
    test_dl: DataLoader,
    loss_function: Callable,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """Test the model on the test dataset.

    Args:
        model (nn.Module): PyTorch module.
        test_dl (DataLoader): Dataloader for the test data.
        loss_function (Callable): Loss function callable.
        device (torch.device, optional): PyTorch device. Defaults to torch.device("cpu").

    Returns:
        tuple[float, float]: Average loss and accuracy for the test data.
    """

    size = len(test_dl.dataset)  # type: ignore
    num_batches = len(test_dl)
    model.eval()

    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in test_dl:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = (correct / size) * 100

    return (test_loss, accuracy)
