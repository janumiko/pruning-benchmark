from timm.models.registry import register_model
from torch import nn
from torchvision import models


@register_model
def resnet18_imagenet1k(**kwargs) -> nn.Module:
    return models.resnet18()
