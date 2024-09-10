from timm.models.registry import register_model
from torch import nn
from transformers import AutoModelForCausalLM


@register_model
def tinystories1m(**kwargs) -> nn.Module:
    return AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
