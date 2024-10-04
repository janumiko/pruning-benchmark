from typing import Any, Iterable

from architecture.pruners.scheluders import BasePruningStepScheduler
import torch
from torch import nn


class BasePruner:
    def __init__(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        pruning_ratio_dict: dict[nn.Module, float],
        pruning_scheduler: BasePruningStepScheduler = None,
        ignored_layers: Iterable[nn.Module] = None,
    ) -> None:
        self.model = model
        self.example_inputs = example_inputs
        self.pruning_ratio_dict = pruning_ratio_dict
        self.ignored_layers = ignored_layers
        self.pruning_scheduler = pruning_scheduler

    def step(self) -> None:
        raise NotImplementedError

    def checkpoint(self) -> dict[str, Any]:
        raise NotImplementedError

    def statistics(self) -> dict[str, float]:
        raise NotImplementedError
