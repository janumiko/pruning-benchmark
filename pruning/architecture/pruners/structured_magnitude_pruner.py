from typing import Any, Callable, Iterable

from architecture.pruners.pruner_base import BasePruner
import torch
from torch import nn
import torch_pruning as tp
import numpy as np


class StructuredMagnitudePruner(BasePruner):
    def __init__(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        pruning_ratio_dict: dict[nn.Module, float],
        importance: Any,
        global_pruning: bool = False,
        ignored_layers: Iterable[nn.Module] = None,
        iterative_steps: int = 1,
        iterative_pruning_ratio_scheduler: Callable[[float, int], list[float]] = None,
    ) -> None:
        super().__init__(
            model=model,
            example_inputs=example_inputs,
            pruning_ratio_dict=pruning_ratio_dict,
            ignored_layers=ignored_layers,
            iterative_steps=iterative_steps,
            iterative_pruning_ratio_scheduler=iterative_pruning_ratio_scheduler,
        )
        self.pruner = tp.MetaPruner(
            model=self.model,
            example_inputs=self.example_inputs,
            pruning_ratio_dict=self.pruning_ratio_dict,
            importance=importance,
            global_pruning=global_pruning,
            ignored_layers=self.ignored_layers,
            iterative_steps=self.pruning_scheduler.steps,
            iterative_pruning_ratio_scheduler=...
        )

        self.base_statistics = self.statistics()

    def step(self) -> None:
        return self.pruner.step()

    def checkpoint(self) -> dict[str, Any]:
        raise NotImplementedError

    def statistics(self) -> dict[str, float]:
        macs, nparams = tp.utils.count_ops_and_params(self.model, self.example_inputs)
        return {
            "macs": macs,
            "nparams": nparams,
        }
    
    def _scheluder(self, pruning_ratio: float, steps: int) -> list[float]:
        return np.cumsum(self.pruning_scheduler())
