from typing import Generator

from architecture.pruners.pruner_base import BasePruner
from architecture.pruners.schedulers import BasePruningSchedule
from architecture.utils import pruning_utils
from architecture.utils.pylogger import RankedLogger
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

logger = RankedLogger(__name__, rank_zero_only=True)


class UnstructuredPruner(BasePruner):
    def __init__(
        self,
        model: nn.Module,
        steps: int,
        pruning_ratio: float,
        pruning_config: dict,
        example_inputs: torch.Tensor,
        pruning_scheduler: BasePruningSchedule,
        global_pruning: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            steps=steps,
            pruning_config=pruning_config,
            example_inputs=example_inputs,
            pruning_scheduler=pruning_scheduler,
        )
        self.pruning_ratio_dict, self.ignored_layers = pruning_utils.parse_prune_config(model, pruning_config)
        self.pruning_ratio = pruning_ratio
        self.current_step = 0

        assert global_pruning, "Global pruning is the only supported method for unstructured pruning"

        if not self.pruning_ratio_dict:
            self.pruning_ratio_dict = {model: pruning_ratio}

        self._pruning_thresholds = self.pruning_scheduler(
            pruning_ratio, self.steps
        )

        logger.info(f"Pruning thresholds: {self._pruning_thresholds}")
        logger.info(f"Ignored layers: {self.ignored_layers}")

        self._params_to_prune = self._get_params_to_prune()
        logger.info(f"Parameters to prune: {self._params_to_prune}")

    def step(self) -> None:
        if self.current_step >= self.steps:
            logger.warning(
                f"Executed more steps than expected: {self.current_step} out of {self.steps}"
            )
            self.current_step += 1
            return

        self._prune()
        self.current_step += 1

    def _prune(self) -> None:
        amount_to_prune = int(
            pruning_utils.calculate_parameters_amount(self._params_to_prune)
            * self._pruning_thresholds[self.current_step]
        )

        pruning_utils.global_unstructured_modified(
            self._params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount_to_prune,
        )

    def _get_unignored_modules(self, module: nn.Module) -> Generator[nn.Module, None, None]:
        for child in module.children():
            if child in self.ignored_layers:
                continue

            yield child
            yield from self._get_unignored_modules(child)

    def _get_params_to_prune(self) -> list[tuple[nn.Module, str]]:
        seen_modules = set()
        params_to_prune = []
        for key in self.pruning_ratio_dict.keys():
            for module in self._get_unignored_modules(key):
                if module in seen_modules:
                    continue
                seen_modules.add(module)

                for name, param in module.named_parameters(recurse=False):
                    if not param.requires_grad:
                        continue
                    params_to_prune.append((module, name))

        return params_to_prune

    def remove(self) -> None:
        for module, name in self._params_to_prune:
            prune.remove(module, name)
