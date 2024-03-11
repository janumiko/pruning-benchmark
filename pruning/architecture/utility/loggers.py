import torch
from .metrics import topk_accuracy
from logging import Logger


class BaseMetricLogger:
    def __init__(self, metric_name: str, logger: Logger) -> None:
        self.metric_name: str = metric_name
        self.logger: Logger = logger
        self.batch_history: list[float | int] = []
        self.epoch_history: list[float | int] = []

    def log_batch(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        raise NotImplementedError

    def log_epoch(self) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the logger to its initial state."""

        self.batch_history = []
        self.epoch_history = []


class AccuracyLogger(BaseMetricLogger):
    def __init__(
        self,
        metric_name: str,
        logger: Logger,
        topk: int = 1,
        is_batch_logging: bool = False,
        is_epoch_logging: bool = False,
    ) -> None:
        super().__init__(metric_name, logger=logger)
        self.topk: int = topk
        self.is_batch_logging: bool = is_batch_logging
        self.is_epoch_logging: bool = is_epoch_logging

    def log_batch(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        """Log the accuracy of the model for a single batch.

        Args:
            prediction (torch.Tensor): Predicted output from the PyTorch module.
            labels (torch.Tensor): Correct labels for the data.
        """
        accuracy = topk_accuracy(predictions, labels, self.topk)
        self.batch_history.append(accuracy)

        if self.is_batch_logging:
            self.logger.info(f"{self.metric_name}: {accuracy:.2f}%")

    def log_epoch(self) -> None:
        """Log the accuracy of the model for a single epoch."""
        epoch_accuracy = sum(self.batch_history) / len(self.batch_history)
        self.epoch_history.append(epoch_accuracy)

        if self.is_epoch_logging:
            self.logger.info(f"{self.metric_name}: {epoch_accuracy:.2f}%")
