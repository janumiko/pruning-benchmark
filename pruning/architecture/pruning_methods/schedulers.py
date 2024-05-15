from typing import Generator

from config.schedulers import BasePruningSchedulerConfig
import numpy as np


class BasePruningStepScheduler:
    def __init__(self, start: float, end: float, step: float) -> None:
        self.start = start
        self.end = end
        self.step = step

    def __iter__(self) -> Generator[float, None, None]:
        raise NotImplementedError


class ConstantStepScheduler(BasePruningStepScheduler):
    def __iter__(self) -> Generator[float, None, None]:
        num_steps = int((self.end - self.start) / self.step)

        if self.start != 0:
            yield self.start

        for _ in range(num_steps):
            yield self.step


class IterativeStepScheduler(BasePruningStepScheduler):
    def __iter__(self) -> Generator[float, None, None]:
        nonpruned_percent = 1

        if self.start != 0:
            yield self.start
            nonpruned_percent -= round(self.start * nonpruned_percent, 8)

        # stop if pruned more than target pruning percentage - 0.1%
        while nonpruned_percent - (1 - self.end) > 0.001:
            current_step = round(self.step * nonpruned_percent, 8)
            nonpruned_percent -= current_step
            nonpruned_percent = round(nonpruned_percent, 8)

            assert current_step > 0, "The pruning step is too small."
            yield current_step


class OneShotStepScheduler(BasePruningStepScheduler):
    def __iter__(self) -> Generator[float, None, None]:
        yield self.end


class LogarithmicStepScheduler(BasePruningStepScheduler):
    def __iter__(self) -> Generator[float, None, None]:
        num_values = int((self.end - self.start) / self.step)
        total_sum = self.end - self.start

        if self.start != 0:
            yield self.start

        values = np.geomspace(1, num_values, num=num_values)

        values *= total_sum / np.sum(values)

        for value in reversed(values):
            yield round(value, 3)


def construct_step_scheduler(
    scheduler_config: BasePruningSchedulerConfig,
) -> BasePruningStepScheduler:
    """Constructs a pruning step scheduler based on the configuration.

    Args:
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
                start=scheduler_config.start,
                end=scheduler_config.end,
                step=scheduler_config.step,
            )
        case "one-shot":
            return OneShotStepScheduler(
                start=scheduler_config.start,
                end=scheduler_config.end,
                step=scheduler_config.step,
            )
        case "logarithmic":
            return LogarithmicStepScheduler(
                start=scheduler_config.start,
                end=scheduler_config.end,
                step=scheduler_config.step,
            )
        case "constant":
            return ConstantStepScheduler(
                start=scheduler_config.start,
                end=scheduler_config.end,
                step=scheduler_config.step,
            )
        case _:
            raise ValueError(f"Unknown scheduler type: {scheduler_config.name}")
