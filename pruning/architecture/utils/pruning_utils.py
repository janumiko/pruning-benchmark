from typing import Mapping

import torch.nn as nn


def parse_prune_config(
    model: nn.Module, config: dict, parent_name: str = ""
) -> dict[nn.Module, float]:
    layers = {}
    if "prune_rate" in config:
        layers[model] = config["prune_rate"]

    for layer_name, prune_info in config.items():
        full_layer_name = f"{parent_name}.{layer_name}" if parent_name else layer_name

        if isinstance(layer_name, int):
            layer = model[layer_name]
        else:
            layer = getattr(model, layer_name, None)

        if isinstance(prune_info, Mapping):
            if "prune_rate" in prune_info:
                if layer is not None:
                    layers[layer] = prune_info["prune_rate"]
            sub_layers = parse_prune_config(layer, prune_info, full_layer_name)
            layers.update(sub_layers)
    return layers
