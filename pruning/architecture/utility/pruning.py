import torch
import torch.nn as nn
from typing import Iterable


def get_parameters_to_prune(model: nn.Module) -> list[nn.Parameter]:
    return [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, nn.Conv2d | nn.Linear)
    ]


def calculate_total_sparsity(
    module: nn.Module, parameters_to_prune: Iterable[tuple[nn.Module, str]]
) -> float:
    total_weights = 0
    total_zero_weights = 0

    pruned_parameters: set[tuple[nn.Module, str]] = set(parameters_to_prune)

    for _, module in module.named_children():
        for param_name, param in module.named_parameters():
            if (module, param_name) not in pruned_parameters:
                continue

            if "weight" in param_name:
                total_weights += float(param.nelement())
                total_zero_weights += float(torch.sum(param == 0))

    sparsity = 100.0 * total_zero_weights / total_weights
    return sparsity


def calculate_parameters_amount(modules: Iterable[tuple[nn.Module, str]]) -> int:
    """Calculate the total amount of parameters in a list of modules.

    Args:
        modules (Iterable[tuple[nn.Module, str]]): List of modules and the parameter names.

    Returns:
        int: The total amount of parameters.
    """

    total_parameters = 0
    for module, parameter in modules:
        for param_name, param in module.named_parameters():
            if param_name == parameter:
                total_parameters += param.nelement()

    return total_parameters
