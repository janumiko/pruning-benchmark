from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class PruningIterator:
    name: str = MISSING
    start: float = MISSING
    end: float = MISSING
    step: float = MISSING


@dataclass
class IterativePruningStrategy(PruningIterator):
    name: str = "iterative"
    start: float = 0.0
    end: float = MISSING
    step: float = MISSING


@dataclass
class OneShotPruningStrategy(PruningIterator):
    name: str = "one-shot"
    start: float = 0.0
    end: float = MISSING
    step: float = 0.0


@dataclass
class LogarithmicPruningStrategy(PruningIterator):
    name: str = "logarithmic"
    start: float = 0.0
    end: float = MISSING
    step: float = MISSING
