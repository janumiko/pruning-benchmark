from typing import Generator

from config.iterators import PruningIterator
import numpy as np


def iterative(start: float, end: float, step: float) -> Generator[float, None, None]:
    """Generates a sequence of numbers in an iterative manner.

    Args:
        start (float): The starting value of the sequence.
        end (float): The end value of the sequence.
        step (float): The step size between consecutive values.

    Yields:
        float: The next value in the sequence.

    """
    num_steps = int((end - start) / step) - 1
    for _ in np.linspace(start, end - step, num_steps):
        yield step


def one_shot(start: float, end: float, step: float) -> Generator[float, None, None]:
    """Generates a single value and yields it.

    Args:
        start (float): The starting value.
        end (float): The ending value.
        step (float): The step size.

    Yields:
        float: The end value.
    """
    yield end


def logarithmic(start: float, end: float, step: float) -> Generator[float, None, None]:
    """Generates logarithmically spaced values between a start and end value.

    Args:
        start (float): The starting value.
        end (float): The ending value.
        step (float): The step size.

    Yields:
        float: The next logarithmically spaced value.
    """
    num_values = int((end - start) / step)
    total_sum = end - start

    values = np.logspace(np.log10(start), np.log10(num_values), num=num_values, base=10)

    values *= total_sum / np.sum(values)

    yield start
    for value in reversed(values):
        yield round(value, 3)


def get_iterator(iterator: PruningIterator) -> Generator[float, None, None]:
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
    match iterator.name:
        case "iterative":
            return iterative(iterator.start, iterator.end, iterator.step)
        case "one-shot":
            return one_shot(iterator.start, iterator.end, iterator.step)
        case "logarithmic":
            return logarithmic(iterator.start, iterator.end, iterator.step)
        case _:
            raise ValueError(f"Unknown iterator type: {iterator.name}")
