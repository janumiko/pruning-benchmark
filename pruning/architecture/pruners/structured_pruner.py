from typing import Any, Callable

from architecture.pruners.pruner_base import BasePruner
from architecture.pruners.schedulers import BasePruningSchedule
import numpy as np
import torch
from torch import nn
import torch_pruning as tp
from architecture.utils import pruning_utils
from architecture.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class StructuredPruner(BasePruner):
    def __init__(
        self,
        model: nn.Module,
        pruning_config: dict,
        steps: int,
        example_inputs: torch.Tensor,
        pruning_scheduler: BasePruningSchedule,
        importance: Callable,
        global_pruning: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            pruning_config=pruning_config,
            example_inputs=example_inputs,
            pruning_scheduler=pruning_scheduler,
            steps=steps,
        )
        self.pruning_ratio_dict, self.ignored_layers = pruning_utils.parse_prune_config(
            model, pruning_config
        )

        self._pruner = tp.MetaPruner(
            model=self.model,
            example_inputs=self.example_inputs,
            pruning_ratio_dict=self.pruning_ratio_dict,
            importance=importance,
            global_pruning=global_pruning,
            ignored_layers=self.ignored_layers,
            iterative_steps=self.steps,
            iterative_pruning_ratio_scheduler=self._scheluder,
        )

        logger.info(f"Ignored layers: {self.ignored_layers}")

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
        x = [0] + np.cumsum(
            self.pruning_scheduler(target_sparsity=pruning_ratio, steps=steps)
        ).tolist()
        return x
