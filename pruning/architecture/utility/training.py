import logging
from pathlib import Path
import random
from typing import Callable, Generator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train_epoch(
    module: nn.Module,
    train_dl: DataLoader,
    optimizer: optim.Optimizer,
    loss_function: Callable,
    device: torch.device,
) -> float:
    """Train a module for one epoch.

    Args:
        model (nn.Module): A PyTorch module.
        train_dl (DataLoader): Dataloader for the train data.
        optimizer (optim.Optimizer): Optimizer instance.
        loss_function (Callable): Loss function callable.
        device (torch.device, optional): Pytorch device.

    Returns:
        float: Average loss for the epoch.
    """

    module.train()
    train_loss = 0

    total_samples = len(train_dl.dataset)
    processed_samples = 0
    log_batch_iter = max(1, len(train_dl) // 10)

    for batch, (inputs, labels) in enumerate(train_dl):
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
        processed_samples += len(inputs)
        if batch % log_batch_iter == 0:
            logger.debug(
                f"Processed {processed_samples}/{total_samples} samples, loss: {loss.item():.6f}"
            )

    return train_loss / len(train_dl)


def validate_epoch(
    module: nn.Module,
    valid_dl: DataLoader,
    loss_function: Callable,
    metrics_functions: Mapping[str, Callable],
    device: torch.DeviceObjType,
) -> dict[str, float]:
    """Validate the model on given data.
    Args:
        model (nn.Module): PyTorch module.
        valid_dl (DataLoader): Dataloader for the validation data.
        loss_function (Callable): Loss function callable.
        metrics_functions (Mapping[str, Callable]): Dictionary with metric_name : callable pairs.
        enable_autocast (bool, optional): Whether to use automatic mixed precision. Defaults to True.
        device (torch.device, optional): Pytorch device.
    Returns:
        dict[str, float]: Dictionary with average loss and metrics for the validation data.
    """

    module.eval()
    metrics = {name: 0.0 for name in metrics_functions.keys()}
    metrics["validation_loss"] = 0.0

    total_samples = len(valid_dl.dataset)
    processed_samples = 0
    log_batch_iter = max(1, len(valid_dl) // 10)

    with torch.no_grad():
        for batch, (X, labels) in enumerate(valid_dl):
            X, labels = X.to(device), labels.to(device)
            # Compute prediction error
            pred = module(X)
            loss = loss_function(pred, labels)

            metrics["validation_loss"] += loss.item()
            for name, func in metrics_functions.items():
                metrics[name] += func(pred, labels)

            processed_samples += len(X)

            if batch % log_batch_iter == 0:
                logger.debug(
                    f"Processed {processed_samples}/{total_samples} samples, loss: {loss.item():.6f}"
                )

    for name, _ in metrics.items():
        metrics[name] /= len(valid_dl)

    return metrics


def setup_ddp(rank: int, world_size: int, init_method: str, seed: int = None) -> None:
    """Setup the distributed training environment.

    Args:
        rank (int): The rank of the process.
        world_size (int): The number of processes.
        init_method (str): The initialization method.
        seed (int, optional): The seed for reproducibility.
    """

    if seed is not None:
        set_reproducibility(seed)

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    dist.barrier()


def cleanup_ddp() -> None:
    """Cleanup the distributed training environment."""
    dist.barrier()
    dist.destroy_process_group()


def gather_metrics(metrics: dict[str, float], world_size: int) -> dict[str, float]:
    """Gather metrics from all processes.

    Args:
        metrics (dict[str, float]): Metrics dictionary.
        world_size (int): Number of processes.

    Returns:
        dict[str, float]: Gathered metrics dictionary.
    """
    gathered_metrics = {}

    for key in metrics.keys():
        tensor = torch.tensor(metrics[key]).cuda()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        gathered_metrics[key] = tensor.item() / world_size

    return gathered_metrics


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


def create_output_csv(path: Path, columns: Sequence[str]) -> None:
    """Create a CSV file with the given columns

    Args:
        path (str): The path to the CSV file
        columns (Sequence[str]): The columns for the CSV file
    """
    df = pd.DataFrame(columns=columns)
    df.to_csv(path, index=False, mode="w")


class EarlyStopper:
    """Early stopping class.
    Monitors a given metric and stops training if it does not improve after a given patience.
    Patience is the number of epochs to wait for improvement before stopping.
    """

    def __init__(self, patience: int, min_delta: float, is_decreasing: bool) -> None:
        """Initialize EarlyStopper.

        Args:
            patience (int): Number of epochs to wait for improvement.
            min_delta (float): Delta which which is used to decide epoch as improvement.
            is_decreasing (bool): Whether the metric should be decreasing or increasing to be considered as improvement.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.is_decreasing = is_decreasing
        self.best_metric_value = float("inf") if is_decreasing else float("-inf")

    def check_stop(self, metric_value: float) -> bool:
        """Check if training should be stopped.

        Args:
            metric_value (float): metric on which to check for improvement.
        Returns:
            bool: Boolean indicating whether training should be stopped.
        """

        if self.is_decreasing:
            if metric_value < self.best_metric_value - self.min_delta:
                self.best_metric_value = metric_value
                self.counter = 0
            else:
                self.counter += 1
        else:
            if metric_value > self.best_metric_value + self.min_delta:
                self.best_metric_value = metric_value
                self.counter = 0
            else:
                self.counter += 1

        return self.counter >= self.patience

    def reset(self) -> None:
        """Reset the early stopper"""
        self.counter = 0
        self.best_metric_value = float("inf") if self.is_decreasing else float("-inf")


def construct_patience_generator(patiences: list[int]) -> Generator[int, None, None]:
    """Construct a generator that repeats the given patiences.
    Yield from patiences but if it ends empty the last value infinitely

    Args:
        patiences (list[int]): List of patiences.

    Yields:
        Generator[int, None, None]: Generator which generates patience values.
    """

    yield from patiences
    while True:
        yield patiences[-1]
