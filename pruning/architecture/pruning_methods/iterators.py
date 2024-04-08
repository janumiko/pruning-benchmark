from typing import Generator

from config.iterators import PruningIterator
import numpy as np


class SequenceGeneratorBase:
    def __init__(self, start: float, end: float, step: float) -> None:
        self.start = start
        self.end = end
        self.step = step

    def __iter__(self) -> Generator[float, None, None]:
        raise NotImplementedError


class LinearSequenceGenerator(SequenceGeneratorBase):
    def __iter__(self) -> Generator[float, None, None]:
        num_steps = int((self.end - self.start) / self.step) - 1

        if self.start != 0:
            yield self.start

        for _ in np.linspace(self.start, self.end, num_steps):
            yield self.step


class OneShotSequenceGenerator(SequenceGeneratorBase):
    def __iter__(self) -> Generator[float, None, None]:
        yield self.end


class LogarithmicSequenceGenerator(SequenceGeneratorBase):
    def __iter__(self) -> Generator[float, None, None]:
        num_values = int((self.end - self.start) / self.step)
        total_sum = self.end - self.start

        if self.start != 0:
            num_values += 1
            yield self.start

        values = np.logspace(np.log10(self.start), np.log10(num_values), num=num_values, base=10)

        values *= total_sum / np.sum(values)

        for value in reversed(values):
            yield round(value, 3)


def construct_iterator(iterator: PruningIterator) -> SequenceGeneratorBase:
    """
    Returns an iterator based on the specified name.

    Args:
        name (str): The name of the iterator type.
        start (float): The starting value of the iterator.
        end (float): The ending value of the iterator.
        step (float): The step size of the iterator.

    Returns:
        Generator[float, None, None]: An iterator that generates floating-point values.

    Raises:
        ValueError: If the specified iterator type is unknown.
    """
    assert iterator.end < 1, "The pruning iterator end value must be less than 1"

    match iterator.name:
        case "iterative":
            return LinearSequenceGenerator(iterator.start, iterator.end, iterator.step)
        case "one-shot":
            return OneShotSequenceGenerator(iterator.start, iterator.end, iterator.step)
        case "logarithmic":
            return LogarithmicSequenceGenerator(iterator.start, iterator.end, iterator.step)
        case _:
            raise ValueError(f"Unknown iterator type: {iterator.name}")
