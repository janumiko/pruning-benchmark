from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class PruningIterator:
    name: str = MISSING
    start: float = MISSING
    end: float = MISSING
    step: float = MISSING


@dataclass
class Iterative(PruningIterator):
    name: str = "iterative"
    start: float = 0.0
    end: float = 1.0
    step: float = MISSING


@dataclass
class OneShot(PruningIterator):
    name: str = "one-shot"
    start: float = 0.0
    end: float = MISSING
    step: float = 0.0

