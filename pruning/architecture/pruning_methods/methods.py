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

    for i, (module, name) in enumerate(parameters_to_prune):
        match module:
            case nn.Conv2d | nn.Linear:
                pruning_amount = int(module._parameters[name].numel() * prune_percent)
                prune.ln_structured(module, name, norm=norm, dim=dim, amount=pruning_amount)

                # apply the same mask to the corresponding batch norm layer
                for module, _ in parameters_to_prune[i : i + 5]:
                    if not isinstance(module, nn.BatchNorm2d):
                        continue

                    layer_mask = module.weight_mask
                    batch_norm_mask = ~torch.all(
                        layer_mask.view(layer_mask.size(0), -1) == 0, dim=1
                    )
                    prune.custom_from_mask(
                        parameters_to_prune[i + 1][0], name="weight", mask=batch_norm_mask
                    )
                    break
            case nn.BatchNorm2d:
                continue
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
