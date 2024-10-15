# train/metrics.py

import logging
from typing import Optional, List

import torch

logger = logging.getLogger(__name__)


def binarize_predictions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Binarize the predictions based on the specified threshold.

    Parameters
    ----------
    logits : torch.Tensor
        Predicted logits (before sigmoid), shape (N, C, H, W).
    threshold : float, optional
        Threshold for binarization. Defaults to 0.5.

    Returns
    -------
    torch.Tensor
        Binarized predictions, shape (N, C, H, W).
    """
    probs = torch.sigmoid(logits)
    binarized = (probs >= threshold).float()
    return binarized


def _check_shapes(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.shape != targets.shape:
        raise ValueError(f"Logits and targets must have the same shape, got {logits.shape} and {targets.shape}.")


class Metric:
    """
    Base class for all metrics.
    """

    def reset(self) -> None:
        """
        Reset the internal state of the metric.
        """
        raise NotImplementedError

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update the metric state with new predictions and targets.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits (before sigmoid), shape (N, C, H, W).
        targets : torch.Tensor
            Ground truth binary masks, shape (N, C, H, W).
        """
        raise NotImplementedError

    def compute(self) -> float:
        """
        Compute the metric value based on the current state.

        Returns
        -------
        float
            The computed metric value.
        """
        raise NotImplementedError


class Accuracy(Metric):
    """
    Compute the binary accuracy.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        _check_shapes(logits, targets)
        preds = binarize_predictions(logits, self.threshold)
        self.correct += (preds == targets).sum().item()
        self.total += targets.numel()

    def compute(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class Precision(Metric):
    """
    Compute the binary precision.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.true_positive = 0
        self.false_positive = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        _check_shapes(logits, targets)
        preds = binarize_predictions(logits, self.threshold)
        self.true_positive += ((preds == 1) & (targets == 1)).sum().item()
        self.false_positive += ((preds == 1) & (targets == 0)).sum().item()

    def compute(self) -> float:
        denominator = self.true_positive + self.false_positive
        if denominator == 0:
            return 0.0
        return self.true_positive / denominator


class Recall(Metric):
    """
    Compute the binary recall.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.true_positive = 0
        self.false_negative = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        _check_shapes(logits, targets)
        preds = binarize_predictions(logits, self.threshold)
        self.true_positive += ((preds == 1) & (targets == 1)).sum().item()
        self.false_negative += ((preds == 0) & (targets == 1)).sum().item()

    def compute(self) -> float:
        denominator = self.true_positive + self.false_negative
        if denominator == 0:
            return 0.0
        return self.true_positive / denominator


class F1Score(Metric):
    """
    Compute the binary F1 Score.
    """

    def __init__(self, threshold: float = 0.5):
        self.precision_metric = Precision(threshold)
        self.recall_metric = Recall(threshold)

    def reset(self) -> None:
        self.precision_metric.reset()
        self.recall_metric.reset()

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        self.precision_metric.update(logits, targets)
        self.recall_metric.update(logits, targets)

    def compute(self) -> float:
        precision = self.precision_metric.compute()
        recall = self.recall_metric.compute()
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


class IoU(Metric):
    """
    Compute the Intersection over Union (IoU) score.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.intersection = 0
        self.union = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        _check_shapes(logits, targets)
        preds = binarize_predictions(logits, self.threshold)
        self.intersection += ((preds == 1) & (targets == 1)).sum().item()
        self.union += ((preds == 1) | (targets == 1)).sum().item()

    def compute(self) -> float:
        if self.union == 0:
            return 0.0
        return self.intersection / self.union


class DiceCoefficient(Metric):
    """
    Compute the Dice Coefficient.
    """

    def __init__(self, threshold: float = 0.5, smooth: float = 1.0):
        self.threshold = threshold
        self.smooth = smooth
        self.reset()

    def reset(self) -> None:
        self.intersection = 0.0
        self.sum_preds = 0.0
        self.sum_targets = 0.0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        _check_shapes(logits, targets)
        preds = binarize_predictions(logits, self.threshold)
        self.intersection += (preds * targets).sum().item()
        self.sum_preds += preds.sum().item()
        self.sum_targets += targets.sum().item()

    def compute(self) -> float:
        denominator = self.sum_preds + self.sum_targets + self.smooth
        if denominator == 0:
            return 0.0
        return (2 * self.intersection + self.smooth) / denominator


def get_metrics(names: Optional[List[str]] = None, **kwargs) -> List[Metric]:
    """
    Factory function to retrieve a list of metric instances based on names.

    Parameters
    ----------
    names : List[str], optional
        List of metric names to instantiate.
        Options include 'accuracy', 'precision', 'recall', 'f1', 'iou', 'dice'.
        If None, defaults to ['accuracy', 'precision', 'recall', 'f1', 'iou', 'dice'].
    **kwargs
        Additional keyword arguments to pass to the metric constructors.

    Returns
    -------
    List[Metric]
        List of metric instances.

    Raises
    ------
    ValueError
        If an unsupported metric name is provided.
    """
    if names is None:
        names = ['accuracy', 'precision', 'recall', 'f1', 'iou', 'dice']

    metric_classes = {
        'accuracy': Accuracy,
        'precision': Precision,
        'recall': Recall,
        'f1': F1Score,
        'iou': IoU,
        'dice': DiceCoefficient,
    }

    metrics = []
    for name in names:
        key = name.lower()
        if key in metric_classes:
            metrics.append(metric_classes[key](**kwargs))
        else:
            raise ValueError(f"Unsupported metric: {name}")
    return metrics
