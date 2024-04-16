from typing import Generator, Iterable

from architecture.utility.pruning import calculate_parameters_amount
from config.schedulers import BasePruningSchedulerConfig
import numpy as np
from torch import nn


class BasePruningStepScheduler:
    def __init__(
        self, pruned_modules: Iterable[nn.Module], start: float, end: float, step: float
    ) -> None:
        self.start = start
        self.end = end
        self.step = step
        self.inital_param_count = calculate_parameters_amount(pruned_modules)

    def __iter__(self) -> Generator[int, None, None]:
        raise NotImplementedError


class ConstantStepScheduler(BasePruningStepScheduler):
    def __init__(
        self, pruned_modules: list[nn.Module], start: float, end: float, step: float
    ) -> None:
        super().__init__(pruned_modules, start, end, step)

    def __iter__(self) -> Generator[int, None, None]:
        num_steps = int((self.end - self.start) / self.step)

        if self.start != 0:
            yield int(self.start * self.inital_param_count)

        for _ in range(num_steps):
            yield int(self.step * self.inital_param_count)


class IterativeStepScheduler(BasePruningStepScheduler):
    def __iter__(self) -> Generator[int, None, None]:
        if self.start != 0:
            yield int(self.start * self.inital_param_count)

        pruned_count = int(self.start * self.inital_param_count)
        current_step = 0
        while pruned_count < self.end * self.inital_param_count:
            current_step = int(self.step * (self.inital_param_count - pruned_count))

            assert current_step <= 0, "The pruning step is too small."

            pruned_count += current_step
            yield current_step


class OneShotStepScheduler(BasePruningStepScheduler):
    def __iter__(self) -> Generator[int, None, None]:
        yield int(self.end * self.inital_param_count)


class LogarithmicStepScheduler(BasePruningStepScheduler):
    def __iter__(self) -> Generator[int, None, None]:
        num_values = int((self.end - self.start) / self.step)
        total_sum = self.end - self.start

        if self.start != 0:
            num_values -= 1
            yield int(self.start * self.inital_param_count)

        values = np.geomspace(1, num_values, num=num_values)

        values *= total_sum / np.sum(values)

        for value in reversed(values):
            yield int(round(value, 3) * self.inital_param_count)


def construct_step_scheduler(
    module: nn.Module,
    scheduler_config: BasePruningSchedulerConfig,
) -> BasePruningStepScheduler:
    """Constructs a pruning step scheduler based on the configuration.

    Args:
        module (nn.Module): The model to prune.
        scheduler_config (BasePruningSchedulerConfig): Configuration for the pruning scheduler.

    Raises:
        ValueError: If the scheduler type is unknown.

    Returns:
        BasePruningStepScheduler: Pruning step scheduler.
    """
    assert scheduler_config.end < 1, "The pruning iterator end value must be less than 1"

    match scheduler_config.name:
        case "iterative":
            return IterativeStepScheduler(
                module,
                start=scheduler_config.start,
                end=scheduler_config.end,
                step=scheduler_config.step,
            )
        case "one-shot":
            return OneShotStepScheduler(
                module,
                start=scheduler_config.start,
                end=scheduler_config.end,
                step=scheduler_config.step,
            )
        case "logarithmic":
            return LogarithmicStepScheduler(
                module,
                start=scheduler_config.start,
                end=scheduler_config.end,
                step=scheduler_config.step,
            )
        case "constant":
            return ConstantStepScheduler(
                module,
                start=scheduler_config.start,
                end=scheduler_config.end,
                step=scheduler_config.step,
            )
        case _:
            raise ValueError(f"Unknown scheduler type: {scheduler_config.name}")
