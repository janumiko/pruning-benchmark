import torch
import torch.nn as nn
from typing import Iterable


def get_parameters_to_prune(model: nn.Module) -> list[nn.Parameter]:
    return [
        (module, "weight")
        for module in model.modules()
        if any(param.requires_grad for param in module.parameters())
        and len(list(module.children())) == 0
    ]


def calculate_total_sparsity(
    model: nn.Module, parameters_to_prune: Iterable[tuple[nn.Module, str]]
) -> float:
    total_weights = 0
    total_zero_weights = 0

    pruned_parameters: set[tuple[nn.Module, str]] = set(parameters_to_prune)

    for module in model.modules():
        for param_name, param in module.named_parameters():
            if (module, param_name) not in pruned_parameters:
                continue

            if "weight" in param_name:
                total_weights += float(param.nelement())
                total_zero_weights += float(torch.sum(param == 0))

    return total_zero_weights / total_weights


def calculate_parameters_amount(modules: Iterable[tuple[nn.Module, str]]) -> int:
    """Calculate the total amount of parameters in a list of modules.

    Args:
        modules (Iterable[tuple[nn.Module, str]]): List of modules and the parameter names.

    Returns:
        int: The total amount of parameters.
    """

    total_parameters = 0
    for module, _ in modules:
        for param in module.parameters():
            total_parameters += param.nelement()

    return total_parameters
