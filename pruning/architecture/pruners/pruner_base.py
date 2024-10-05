from typing import Any, Iterable

from architecture.pruners.scheluders import BasePruningScheduler
from architecture.utils import pruning_utils
import torch
from torch import nn


class BasePruner:
    def __init__(
        self,
        model: nn.Module,
        pruning_config: dict,
        example_inputs: torch.Tensor,
        pruning_scheduler: BasePruningScheduler,
        ignored_layers: Iterable[nn.Module] = None,
    ) -> None:
        self.model = model
        self.pruning_config = pruning_config
        self.example_inputs = example_inputs
        self.ignored_layers = ignored_layers
        self.pruning_scheduler = pruning_scheduler
        self.pruning_ratio_dict = pruning_utils.parse_prune_config(model, pruning_config)
        print(self.pruning_ratio_dict)

    @property
    def scheduler_steps(self) -> int:
        return self.pruning_scheduler.steps

    def step(self) -> None:
        raise NotImplementedError

    def checkpoint(self) -> dict[str, Any]:
        raise NotImplementedError

    def statistics(self) -> dict[str, float]:
        raise NotImplementedError
