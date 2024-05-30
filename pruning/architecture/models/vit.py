from timm import create_model
from timm.models.registry import register_model


@register_model
def vit_small_patch16_224(num_classes: int, **kwargs):
    model = create_model("vit_small_patch16_224", num_classes=num_classes)

    return model
