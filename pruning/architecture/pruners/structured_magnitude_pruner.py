from typing import Any, Callable, Iterable

from architecture.pruners.pruner_base import BasePruner
from architecture.pruners.scheluders import BasePruningScheduler
import torch
from torch import nn
import torch_pruning as tp
import numpy as np


class StructuredMagnitudePruner(BasePruner):
    def __init__(
        self,
        model: nn.Module,
        pruning_config: dict,
        example_inputs: torch.Tensor,
        pruning_scheduler: BasePruningScheduler,
        importance: Callable,
        ignored_layers: Iterable[nn.Module] = None,
        global_pruning: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            pruning_config=pruning_config,
            example_inputs=example_inputs,
            ignored_layers=ignored_layers,
            pruning_scheduler=pruning_scheduler,
        )
        self._pruner = tp.MetaPruner(
            model=self.model,
            example_inputs=self.example_inputs,
            pruning_ratio_dict=self.pruning_ratio_dict,
            importance=importance,
            global_pruning=global_pruning,
            ignored_layers=self.ignored_layers,
            iterative_steps=self.pruning_scheduler.steps,
            iterative_pruning_ratio_scheduler=self._scheluder,
        )

        self.base_statistics = self.statistics()

    def step(self) -> None:
        return self._pruner.step()

    def checkpoint(self) -> dict[str, Any]:
        raise NotImplementedError

    def statistics(self) -> dict[str, float]:
        macs, nparams = tp.utils.count_ops_and_params(self.model, self.example_inputs)
        return {
            "macs": macs,
            "nparams": nparams,
        }

    def _scheluder(self, pruning_ratio: float, steps: int) -> list[float]:
        x = [0] + np.cumsum(self.pruning_scheduler(target_sparsity=pruning_ratio, steps=steps)).tolist()
        return x
