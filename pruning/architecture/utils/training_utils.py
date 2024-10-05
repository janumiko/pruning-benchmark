import random
from typing import Generator, Literal
from architecture.utils.pylogger import RankedLogger

import numpy as np
import torch


logger = RankedLogger(__name__, rank_zero_only=True)


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


class RestoreCheckpoint:
    def __init__(
        self,
        enabled: bool,
        monitor: str,
        min_delta: float,
        mode: Literal["min", "max"],
    ):
        self.enabled = enabled
        self.monitor = monitor
        self.min_delta = min_delta
        self.mode = mode

        self.best_metric_value = float("inf") if mode == "min" else float("-inf")
        self._state_dict = None

    def update(self, model: torch.nn.Module, results: dict[str, float]) -> None:
        metric_value = results[self.monitor]
        if self.mode == "min":
            update = metric_value < self.best_metric_value - self.min_delta
        else:
            update = metric_value > self.best_metric_value + self.min_delta

        if update:
            logger.info(f"Updating best model with metric {self.monitor}: {metric_value}")
            self.best_metric_value = metric_value
            self._state_dict = model.state_dict()

    def restore_best(self, model: torch.nn.Module) -> None:
        if self.enabled and self._state_dict is not None:
            model.load_state_dict(self._state_dict)

    def reset(
        self, init_results: dict[str, float] | None = None, state_dict: dict | None = None
    ) -> None:
        if init_results is not None:
            self.best_metric_value = init_results[self.monitor]
        else:
            self.best_metric_value = float("inf") if self.mode == "min" else float("-inf")
        self._state_dict = state_dict


class EarlyStopper:
    """Early stopping class.
    Monitors a given metric and stops training if it does not improve after a given patience.
    Patience is the number of epochs to wait for improvement before stopping.
    """

    def __init__(
        self,
        enabled: bool,
        monitor: str,
        patience: int | list[int],
        min_delta: float,
        mode: Literal["min", "max"],
        overide_epochs_to_inf: bool = False,
        **kwargs,
    ):
        self.enabled = enabled
        self.monitor = monitor
        self.patience_gen = self._patience_gen(patience)
        self.current_patience = next(self.patience_gen)
        self.min_delta = min_delta
        self.mode = mode
        self.overide_epochs_to_inf = overide_epochs_to_inf

        self.counter = 0
        self.best_metric_value = float("inf") if mode == "min" else float("-inf")

    def _patience_gen(self, patience: int | list[int]) -> Generator[int, None, None]:
        if not isinstance(patience, int):
            yield from patience
        while True:
            yield patience

    def check_stop(self, results: dict[str, float]) -> bool:
        """Check if the training should stop.
        Args:
            results (dict): Dictionary with the results of the validation loop.
        Returns:
            bool: True if the training should stop, False otherwise.
        """
        if not self.enabled:
            return False

        metric_value = results[self.monitor]
        if self.mode == "min":
            stop = metric_value < self.best_metric_value - self.min_delta
        else:
            stop = metric_value > self.best_metric_value + self.min_delta

        if stop:
            logger.info(f"Improvement in {self.monitor}: {metric_value}")
            self.best_metric_value = metric_value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.current_patience

    def reset(self, init_results: dict[str, float] | None = None, next_patience: bool = False) -> None:
        self.counter = 0
        if init_results is not None:
            self.best_metric_value = init_results[self.monitor]
        else:
            self.best_metric_value = float("inf") if self.mode == "min" else float("-inf")

        if next_patience:
            self.current_patience = next(self.patience_gen)
