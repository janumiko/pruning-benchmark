# metrics for early stopping.

from dataclasses import dataclass


@dataclass
class BaseMetric:
    name: str
    is_decreasing: bool = False
    _default_value: float = float("inf" if is_decreasing else "-inf")


@dataclass
class Top1Accuracy(BaseMetric):
    name: str = "top1_accuracy"
    is_decreasing: bool = False


@dataclass
class ValidationLoss(BaseMetric):
    name: str = "validation_loss"
    is_decreasing: bool = True


@dataclass
class Top5Accuracy(BaseMetric):
    name: str = "top5_accuracy"
    is_decreasing: bool = False
