import torch

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.247, 0.243, 0.261]

CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]

IMAGENET1K_MEAN = [0.485, 0.456, 0.406]
IMAGENET1K_STD = [0.229, 0.224, 0.225]

LAYER_MAPPING = {
    "Conv1d": torch.nn.Conv1d,
    "Conv2d": torch.nn.Conv2d,
    "Linear": torch.nn.Linear,
    "BatchNorm2d": torch.nn.BatchNorm2d,
    "BatchNorm1d": torch.nn.BatchNorm1d,
    "LayerNorm": torch.nn.LayerNorm,
    "GroupNorm": torch.nn.GroupNorm,
    "InstanceNorm1d": torch.nn.InstanceNorm1d,
    "InstanceNorm2d": torch.nn.InstanceNorm2d,
    "Embedding": torch.nn.Embedding,
}
