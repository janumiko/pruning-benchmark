from timm.models.registry import register_model
from torchvision import models


@register_model
def efficientnetv2s(num_classes: int, **kwargs):
    model = models.efficientnet_v2_s(num_classes=num_classes)

    return model
