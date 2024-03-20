from typing import Iterable
import torch
from torch import nn


def accuracy(output: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate accuracy of the model. An alias for topk_accuracy with k=1.

    Args:
        output (torch.Tensor): Predicted output from the model.
        target (torch.Tensor): Correct labels for the data.

    Returns:
        float: The accuracy of the model in the range [0, 1].
    """
    return topk_accuracy(output, labels, topk=1)


def top5_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate top-5 accuracy of the model. An alias for topk_accuracy with k=5.

    Args:
        prediction (torch.Tensor): Predicted output from the model.
        target (torch.Tensor): Correct labels for the data.

    Returns:
        float: The top-5 accuracy of the model in the range [0, 1].
    """
    return topk_accuracy(prediction, target, topk=5)


def topk_accuracy(predictions: torch.Tensor, labels: torch.Tensor, topk: int) -> float:
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        predpredictionsiction (torch.Tensor): Prediction tensor with shape (batch_size, num_classes).
        labels (torch.Tensor: Ground truth tensor with shape (batch_size).
        topk (int): The values of k to compute the accuracy over.

    Returns:
        float: The top_k accuracy of the model.
    """

    with torch.no_grad():
        top5_pred = torch.topk(predictions, k=topk, dim=1).indices
        assert top5_pred.shape[0] == len(labels)
        correct = 0
        for i in range(topk):
            correct += torch.sum(top5_pred[:, i] == labels).item()

    return (correct * 100) / len(labels)
