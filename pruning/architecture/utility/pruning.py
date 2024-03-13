import torch
import torch.nn as nn
from typing import Iterable


def get_parameters_to_prune(
    model: nn.Module, types_to_prune: tuple[nn.Module]
) -> list[tuple[nn.Module, str]]:
    """Get the parameters to prune from a model.

    Args:
        model (nn.Module): A PyTorch model.
        types_to_prune (tuple[nn.Module]): Tuple of module types to prune. Ex. nn.Linear, nn.Conv2d.

    Returns:
        list[tuple[nn.Module, str]]: List of tuples containing the module and the parameter name.
    """
    return [
        (module, name)
        for module in model.modules()
        if isinstance(module, types_to_prune)
        for name, param in module.named_parameters()
        if param.requires_grad
    ]


def calculate_parameters_sparsity(
    model: nn.Module, parameters_to_prune: Iterable[tuple[nn.Module, str]]
) -> float:
    """Calculate the total sparsity of the model for given parameters.

    Args:
        model (nn.Module): A PyTorch model to calculate the sparsity of.
        parameters_to_prune (Iterable[tuple[nn.Module, str]]): Iterable of parameters which are pruned.

    Returns:
        float: Percentage of zero weights in range from 0 to 100%.
    """
    total_weights = 0
    total_zero_weights = 0

    pruned_parameters: set[tuple[nn.Module, str]] = set(parameters_to_prune)

    for module in model.modules():
        for param_name, param in module.named_parameters():
            if (module, param_name) not in pruned_parameters:
                continue

            total_weights += float(param.nelement())
            total_zero_weights += float(torch.sum(param == 0))

    return (total_zero_weights / total_weights) * 100


def calculate_total_sparsity(model: nn.Module) -> float:
    """Calculate the total sparsity of the model.
    Args:
        model (nn.Module): A PyTorch model to calculate the sparsity of.

    Returns:
        float: Percentage of zero weights in range from 0 to 100%.
    """
    total_weights = 0
    total_zero_weights = 0

    for param in model.parameters():
        if not param.requires_grad:
            continue

        total_weights += float(param.nelement())
        total_zero_weights += float(torch.sum(param == 0))

    return (total_zero_weights / total_weights) * 100


def calculate_parameters_amount(modules: Iterable[tuple[nn.Module, str]]) -> int:
    """Calculate the total amount of parameters in a list of modules.

    Args:
        modules (Iterable[tuple[nn.Module, str]]): List of modules and the parameter names.

    Returns:
        int: The total amount of parameters.
    """

    total_parameters = 0
    for module, param_name in modules:
        total_parameters += module._parameters[param_name].nelement()

    return total_parameters


def get_parameter_count(module: nn.Module) -> int:
    """Calculate the total amount of parameters in a module.

    Args:
        module (nn.Module): A PyTorch module.

    Returns:
        int: The total amount of parameters.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def log_parameters_sparsity(
    model: nn.Module, parameters_to_prune: Iterable[tuple[nn.Module, str]], logger
):
    """Log the sparsity of the model for given parameters.

    Args:
        model (nn.Module): A PyTorch model to calculate the sparsity of.
        parameters_to_prune (Iterable[tuple[nn.Module, str]]): Iterable of parameters which are pruned.
        logger: The logger to use for logging.
    """
    sparsity = 100 - calculate_parameters_sparsity(model, parameters_to_prune)
    logger.info(f"Non-zero weights of the pruned parameters: {sparsity:.2f}%")
    return sparsity


def log_module_sparsity(model: nn.Module, logger):
    """Log the sparsity of the model.

    Args:
        model (nn.Module): A PyTorch model to calculate the sparsity of.
        logger: The logger to use for logging.
    """
    sparsity = 100 - calculate_total_sparsity(model)
    logger.info(f"Non-zero weights of the pruned module: {sparsity:.2f}%")
    return sparsity
