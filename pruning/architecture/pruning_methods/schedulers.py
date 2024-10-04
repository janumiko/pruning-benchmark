import logging
from typing import Generator

from config.schedulers import BasePruningSchedulerConfig

logger = logging.getLogger(__name__)


class BasePruningSchedule:
    def __iter__(self) -> Generator[float, None, None]:
        raise NotImplementedError

    def __call__(self, target_sparsity: float, steps: int) -> list[float]:
        """Constructs a schedule for pruning.

        Args:
            target_sparsity (float): Target sparsity level.
            steps (int): Number of pruning steps.

        Returns:
            list[float]: List with pruning values per step.
        """

        raise NotImplementedError


class ConstantPruningSchedule(BasePruningSchedule):
    def __call__(self, target_sparsity: float, steps: int) -> list[float]:
        """Constructs a constant schedule for pruning.

        Args:
            target_sparsity (float): Target sparsity level.
            steps (int): Number of pruning steps.

        Returns:
            list[float]: List with pruning values per step.
        """

        pruning_step = target_sparsity / steps

        return [pruning_step for _ in range(steps)]


class GeometricPruningSchedule(BasePruningSchedule):
    def __call__(self, target_sparsity: float, steps: int) -> list[float]:
        """Constructs a geometric schedule for pruning.

        Args:
            target_sparsity (float): Target sparsity level.
            steps (int): Number of pruning steps.

        Returns:
            list[float]: List with pruning values per step.
        """

        initial_step = step = 1 - (1 - target_sparsity) ** (1 / steps)

        output_steps = []
        for _ in range(steps):
            step = round(step, 8)
            output_steps.append(step)
            step *= 1 - initial_step

        return output_steps


class OneShotPruningSchedule(BasePruningSchedule):
    def __call__(self, target_sparsity: float, steps: int = 1) -> list[float]:
        """Constructs a one-shot schedule for pruning.

        Args:
            target_sparsity (float): Target sparsity level.

        Returns:
            list[float]: List with pruning values per step.
        """

        if steps > 1:
            raise ValueError("One-shot pruning can only have one step.")

        return [target_sparsity]


class FewShotPruningSchedule(BasePruningSchedule):
    def __init__(self, one_shot_step: float):
        self.one_shot_step = one_shot_step

    def __call__(self, target_sparsity: float, steps: int) -> list[float]:
        """Constructs a few-shot schedule for pruning.

        Args:
            target_sparsity (float): Target sparsity level.
            steps (int): Number of pruning steps.

        Returns:
            list[float]: List with pruning values per step.
        """

        output_steps = [self.one_shot_step]
        initial_step = step = 1 - (1 - (target_sparsity - self.one_shot_step)) ** (1 / steps)

        output_steps = []
        for _ in range(steps):
            step = round(step, 8)
            output_steps.append(step)
            step *= 1 - initial_step

        return output_steps


def construct_step_scheduler(
    scheduler_config: BasePruningSchedulerConfig,
) -> BasePruningSchedule:
    """Constructs a pruning step scheduler based on the configuration.

    Args:
        scheduler_config (BasePruningSchedulerConfig): Configuration for the pruning scheduler.

    Raises:
        ValueError: If the scheduler type is unknown.

    Returns:
        BasePruningScheduler.
    """
    match scheduler_config.name:
        case "geometric":
            return GeometricPruningSchedule()
        case "one-shot":
            return OneShotPruningSchedule()
        case "constant":
            return ConstantPruningSchedule()
        case "few-shot":
            return FewShotPruningSchedule(scheduler_config.one_shot_step)
        case _:
            raise ValueError(f"Unknown scheduler type: {scheduler_config.name}")
