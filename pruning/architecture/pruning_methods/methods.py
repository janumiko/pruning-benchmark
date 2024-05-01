import logging
from typing import Callable, Iterable

from architecture.utility.pruning import calculate_parameters_amount
from config.methods import BasePruningMethodConfig, GlobalL1UnstructuredConfig, LnStructuredConfig
import torch
from torch import nn
import torch.nn.utils.prune as prune

logger = logging.getLogger(__name__)

METHOD_REGISTER = {}


def register_method(method_name: str) -> Callable:
    """Register a pruning method.

    Args:
        method_name (str): The name of the method to register.

    Returns:
        Callable: The decorator function.
    """

    def decorator(func: Callable) -> Callable:
        METHOD_REGISTER[method_name] = func
        return func

    return decorator


@register_method(GlobalL1UnstructuredConfig().name)
def global_l1_unstructured(
    parameters_to_prune: Iterable[tuple[nn.Module, str]],
    prune_percent: float,
    **kwargs,
) -> None:
    """Prune the given parameters using global unstructured pruning with the given pruning method.

    Args:
        parameters_to_prune (Iterable[tuple[nn.Module, str]]): The parameters to prune.
        prune_percent (float): Percent of parameters to prune.
    """

    amount_to_prune = int(calculate_parameters_amount(parameters_to_prune) * prune_percent)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount_to_prune,
    )


@register_method(LnStructuredConfig().name)
def ln_structured(
    parameters_to_prune: Iterable[tuple[nn.Module, str]],
    norm: int | str,
    dim: int,
    prune_percent: float,
    **kwargs,
) -> None:
    """Prune the given parameters using structured pruning with the given pruning method.

    Args:
        parameters_to_prune (Iterable[tuple[nn.Module, str]]): The parameters to prune.
        norm (int | str): The norm to use for pruning.
        dim (int): The dimension to use for pruning.
        prune_percent (float): Percent of parameters to prune.
    """

    prev_module = parameters_to_prune[0][0]
    for module, name in parameters_to_prune:
        if name in module._parameters:
            param_name = name
        else:
            param_name = f"{name}_orig"

        match module:
            case nn.Conv2d() | nn.Linear() if name == "weight":
                channels_to_prune = int(module._parameters[param_name].size(dim) * prune_percent)
                logger.info(
                    f"Pruning {channels_to_prune} channels from {module.__class__.__name__}.{name} dimension {dim} using {norm} norm."
                )
                prune.ln_structured(module, name, n=norm, dim=dim, amount=channels_to_prune)
                prev_module = module
            case nn.Conv2d() | nn.Linear() if name == "bias":
                logger.info(
                    f"Pruning {module.__class__.__name__}.{name} using {prev_module.__class__.__name__}.weight mask."
                )
                prune.custom_from_mask(
                    module,
                    name,
                    torch.all(prev_module.weight_mask.flatten(start_dim=1) == 1, dim=1),
                )
            case nn.BatchNorm2d() if name in {"weight", "bias"}:
                logger.info(
                    f"Pruning {module.__class__.__name__}.{name} using {prev_module.__class__.__name__}.weight mask."
                )
                prune.custom_from_mask(
                    module,
                    name,
                    torch.all(prev_module.weight_mask.flatten(start_dim=1) == 1, dim=1),
                )
            case _:
                logger.warning(f"Unsupported module {module.__class__.__name__}")


def prune_module(
    params: tuple[nn.Module, str], prune_percent: float, pruning_cfg: BasePruningMethodConfig
) -> None:
    """Prune the given module using the given configuration.

    Args:
        params (tuple[nn.Module, str]):
        prune_percent (float): Percent of parameters to prune.
        cfg (BasePruningMethodConfig): Configuration for the pruning method.
    """

    METHOD_REGISTER[pruning_cfg.name](
        parameters_to_prune=params,
        prune_percent=prune_percent,
        **pruning_cfg,
    )
