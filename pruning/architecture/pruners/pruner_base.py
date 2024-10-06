from typing import Any, Iterable

from architecture.pruners.schedulers import BasePruningSchedule
from architecture.utils import pruning_utils
import torch
from torch import nn


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
        self.pruning_ratio_dict, self.ignored_layers = pruning_utils.parse_prune_config(model, pruning_config)
        self.steps = steps
        print(f"Pruning ratio dict: {self.pruning_ratio_dict}")
        print(f"Ignored layers: {self.ignored_layers}")

    def step(self) -> None:
        raise NotImplementedError

    def checkpoint(self) -> dict[str, Any]:
        raise NotImplementedError

    def statistics(self) -> dict[str, float]:
        raise NotImplementedError