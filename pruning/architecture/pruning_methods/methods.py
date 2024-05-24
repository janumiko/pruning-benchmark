import itertools
import logging
from typing import Callable, Iterable, Iterator

from architecture.utility.pruning import calculate_parameters_amount
from config.methods import BasePruningMethodConfig, GlobalL1UnstructuredConfig, LnStructuredConfig
import torch
from torch import nn
import torch.nn.utils.prune as prune

logger = logging.getLogger(__name__)

METHOD_REGISTER = {}


def register_method(
    method_name: str,
) -> Callable[[Iterable[tuple[nn.Module, str]], Iterator[float], dict], None]:
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


def _prune_custom_from_mask(module: nn.Module, name: str, prev_module: nn.Module) -> None:
    """Prune the given module using the mask from the previous module.

    Args:
        module (nn.Module): A PyTorch module.
        name (str): Name of the parameter to prune.
        prev_module (nn.Module): Previous module to get the mask from.
    """
    logger.info(
        f"Pruning {module.__class__.__name__}.{name} using {prev_module.__class__.__name__}.weight mask."
    )

    match module:
        case nn.BatchNorm2d() | nn.Linear():
            mask = torch.all(prev_module.weight_mask.flatten(start_dim=1) == 1, dim=1)
        case nn.Conv2d():
            # TODO add support for Conv2d (shortcut connections)
            pass

    prune.custom_from_mask(module, name, mask)


@register_method(GlobalL1UnstructuredConfig().name)
def global_l1_unstructured(
    parameters_to_prune: Iterable[tuple[nn.Module, str]],
    pruning_values: Iterator[float],
    **kwargs,
) -> None:
    """Prune the given parameters using global unstructured pruning with the given pruning method.

    Args:
        parameters_to_prune (Iterable[tuple[nn.Module, str]]): The parameters to prune.
        pruning_values (Iterator[float]): An iterator of pruning value for every layer.
    """

    amount_to_prune = int(calculate_parameters_amount(parameters_to_prune) * next(pruning_values))

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
    pruning_values: Iterator[float],
    **kwargs,
) -> None:
    """Prune the given parameters using structured pruning with the given pruning method.

    Args:
        parameters_to_prune (Iterable[tuple[nn.Module, str]]): The parameters to prune.
        norm (int | str): The norm to use for pruning.
        dim (int): The dimension to use for pruning.
        pruning_values (Iterator[float]): An iterator of pruning value for every layer.
        kwargs: Catch all the other arguments.
    """

    prev_module = parameters_to_prune[0][0]
    for module, name in parameters_to_prune:
        if name in module._parameters:
            param_name = name
        else:
            param_name = f"{name}_orig"

        match module:
            case nn.Conv2d() if module.kernel_size == (1, 1):
                logger.info(f"Skipping {module.__class__.__name__}.{name} with kernel size 1x1.")
                prune.ln_structured(module, name, n=norm, dim=dim, amount=0)
            case nn.Conv2d() | nn.Linear() if name == "weight":
                channels_to_prune = int(
                    module._parameters[param_name].size(dim) * next(pruning_values)
                )
                logger.info(
                    f"Pruning {channels_to_prune} channels from {module.__class__.__name__}.{name} dimension {dim} using {norm} norm."
                )
                prune.ln_structured(module, name, n=norm, dim=dim, amount=channels_to_prune)
                prev_module = module
            case nn.Conv2d() | nn.Linear() if name == "bias":
                _prune_custom_from_mask(module, name, prev_module)
            case nn.BatchNorm2d() if name in {"weight", "bias"}:
                _prune_custom_from_mask(module, name, prev_module)
            case _:
                logger.warning(f"Unsupported module {module.__class__.__name__}.{name}")


def _create_pruning_iterator(pruning_values: Iterable[float]) -> Iterator[float]:
    """Create an iterator from the list, ensure that the given iterator has multiple values.

    If the iterator has only one value, return a new iterator that repeats that value.

    Args:
        pruning_values (Iterable[float]): The pruning values.

    Returns:
        Iterator[float]: The new iterator.
    """

    if len(pruning_values) == 1:
        return itertools.repeat(pruning_values[0])
    else:
        return iter(pruning_values)


def prune_module(
    params: tuple[nn.Module, str],
    pruning_values: Iterable[float],
    pruning_cfg: BasePruningMethodConfig,
) -> None:
    """Prune the given module using the given configuration.

    Args:
        params (tuple[nn.Module, str]):
        pruning_values (Iterable[float]): An iterator of pruning value for every layer.
        pruning_cfg (BasePruningMethodConfig): Configuration for the pruning method.
    """

    METHOD_REGISTER[pruning_cfg.name](
        parameters_to_prune=params,
        pruning_values=_create_pruning_iterator(pruning_values),
        **pruning_cfg,
    )
