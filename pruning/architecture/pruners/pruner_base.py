from typing import Any

from architecture.pruners.schedulers import BasePruningSchedule
import torch
from torch import nn
from architecture.utils.metrics import BaseMetricLogger


class BasePruner:
    def __init__(
        self,
        model: nn.Module,
        steps: int,
        pruning_config: dict,
        example_inputs: torch.Tensor,
        pruning_scheduler: BasePruningSchedule,
    ) -> None:
        self.model = model
        self.pruning_config = pruning_config
        self.example_inputs = example_inputs
        self.pruning_scheduler = pruning_scheduler
        self.steps = steps

    def step(self) -> None:
        raise NotImplementedError

    def checkpoint(self) -> dict[str, Any]:
        raise NotImplementedError

    def statistics(self) -> dict[str, float]:
        raise NotImplementedError
