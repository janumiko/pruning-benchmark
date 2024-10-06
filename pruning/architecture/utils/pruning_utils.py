from typing import Iterable, Mapping

from config.constants import LAYER_MAPPING
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def parse_prune_config(
    model: nn.Module, config: dict, parent_name: str = ""
) -> tuple[Mapping[nn.Module, float], list[nn.Module]]:
    ignored_types = tuple(
        LAYER_MAPPING[layer_name] for layer_name in config.get("ignored_layers", [])
    )
    model_config: dict = config.get("model", {})

    layers: dict[nn.Module, float] = {}
    ignore_layers: set[nn.Module] = set()

    def should_ignore(layer, prune_rate):
        return prune_rate == 0 or isinstance(layer, ignored_types)

    def process_layer(layer, prune_rate):
        if should_ignore(layer, prune_rate):
            ignore_layers.add(layer)
        else:
            layers[layer] = prune_rate

    def recursive_parse(layer, layer_config, parent_name):
        if "prune_rate" in layer_config:
            process_layer(layer, layer_config["prune_rate"])

        for layer_name, prune_info in layer_config.items():
            if layer_name == "prune_rate":
                continue

            full_layer_name = f"{parent_name}.{layer_name}" if parent_name else layer_name
            sub_layer = (
                layer[layer_name] if isinstance(layer_name, int) else getattr(layer, layer_name, None)
            )

            if isinstance(prune_info, Mapping) and sub_layer is not None:
                prune_rate = prune_info.get("prune_rate")
                if prune_rate is not None:
                    process_layer(sub_layer, prune_rate)
                recursive_parse(sub_layer, prune_info, full_layer_name)

    recursive_parse(model, model_config, parent_name)

    return layers, list(ignore_layers)


def calculate_parameters_amount(modules: Iterable[tuple[nn.Module, str]]) -> int:
    """Calculate the total amount of parameters in a list of modules.

    Args:
        modules (Iterable[tuple[nn.Module, str]]): List of modules and the parameter names.

    Returns:
        int: The total amount of parameters.
    """

    total_parameters = 0
    for module, param_name in modules:
        if param_name in module._parameters:
            param_name = param_name
        else:
            param_name = f"{param_name}_orig"

        total_parameters += module._parameters[param_name].nelement()

    return total_parameters


def global_unstructured_modified(parameters, pruning_method, importance_scores=None, **kwargs):
    r"""
    Modified version of torch.nn.utils.prune.global_l1_unstructured.
    Removes the hooks with masks to reduce memory usage.
    """
    # ensure parameters is a list or generator of tuples
    if not isinstance(parameters, Iterable):
        raise TypeError("global_unstructured(): parameters is not an Iterable")

    importance_scores = importance_scores if importance_scores is not None else {}
    if not isinstance(importance_scores, dict):
        raise TypeError("global_unstructured(): importance_scores must be of type dict")

    # flatten importance scores to consider them all at once in global pruning
    relevant_importance_scores = torch.nn.utils.parameters_to_vector(
        [
            importance_scores.get((module, name), getattr(module, name))
            for (module, name) in parameters
        ]
    )
    # similarly, flatten the masks (if they exist), or use a flattened vector
    # of 1s of the same dimensions as t
    default_mask = torch.nn.utils.parameters_to_vector(
        [
            getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
            for (module, name) in parameters
        ]
    )

    # use the canonical pruning methods to compute the new mask, even if the
    # parameter is now a flattened out version of `parameters`
    container = prune.PruningContainer()
    container._tensor_name = "temp"  # to make it match that of `method`
    method = pruning_method(**kwargs)
    method._tensor_name = "temp"  # to make it match that of `container`
    if method.PRUNING_TYPE != "unstructured":
        raise TypeError(
            'Only "unstructured" PRUNING_TYPE supported for '
            f"the `pruning_method`. Found method {pruning_method} of type {method.PRUNING_TYPE}"
        )

    container.add_pruning_method(method)

    # use the `compute_mask` method from `PruningContainer` to combine the
    # mask computed by the new method with the pre-existing mask
    final_mask = container.compute_mask(relevant_importance_scores, default_mask)

    # Pointer for slicing the mask to match the shape of each parameter
    pointer = 0
    for module, name in parameters:
        param = getattr(module, name)
        # The length of the parameter
        num_param = param.numel()
        # Slice the mask, reshape it
        param_mask = final_mask[pointer : pointer + num_param].view_as(param)
        # Assign the correct pre-computed mask to each parameter and add it
        # to the forward_pre_hooks like any other pruning method
        prune.custom_from_mask(module, name, mask=param_mask)

        for k in list(module._forward_pre_hooks):
            hook = module._forward_pre_hooks[k]
            if isinstance(hook, prune.PruningContainer):
                if isinstance(hook[-1], prune.CustomFromMask):
                    hook[-1].mask = None
            elif isinstance(hook, prune.CustomFromMask):
                hook.mask = None

        # Increment the pointer to continue slicing the final_mask
        pointer += num_param
