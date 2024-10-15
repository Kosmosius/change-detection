# train/metrics.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class MetricError(Exception):
    """Custom exception for metric-related errors."""
    pass


def binarize_predictions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Binarizes the predictions based on the specified threshold.

    Args:
        logits (torch.Tensor): Predicted logits (before sigmoid), shape (N, 1, H, W).
        threshold (float): Threshold for binarization. Defaults to 0.5.

    Returns:
        torch.Tensor: Binarized predictions, shape (N, 1, H, W).

    Raises:
        MetricError: If logits contain NaN or Inf values.
    """
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logger.error("Logits contain NaN or Inf values.")
        raise MetricError("Logits contain NaN or Inf values.")

    probs = torch.sigmoid(logits)
    binarized = (probs >= threshold).float()
    logger.debug(f"Binarized predictions with threshold {threshold}.")
    return binarized


class Metric:
    """
    Base class for all metrics.
    """

    def reset(self) -> None:
        """
        Resets the internal state of the metric.
        """
        raise NotImplementedError

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the metric state with new predictions and targets.

        Args:
            logits (torch.Tensor): Predicted logits (before sigmoid), shape (N, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (N, 1, H, W).
        """
        raise NotImplementedError

    def compute(self) -> float:
        """
        Computes the metric value based on the current state.

        Returns:
            float: The computed metric value.
        """
        raise NotImplementedError


class Accuracy(Metric):
    """
    Computes the binary accuracy.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initializes the Accuracy metric.

        Args:
            threshold (float, optional): Threshold for binarization. Defaults to 0.5.
        """
        self.threshold: float = threshold
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state of the Accuracy metric.
        """
        self.correct: int = 0
        self.total: int = 0
        logger.debug("Accuracy metric state has been reset.")

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the Accuracy metric with new predictions and targets.

        Args:
            logits (torch.Tensor): Predicted logits (before sigmoid), shape (N, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (N, 1, H, W).

        Raises:
            MetricError: If logits and targets have mismatched shapes.
        """
        if logits.shape != targets.shape:
            logger.error(f"Shape mismatch: logits shape {logits.shape} vs targets shape {targets.shape}")
            raise MetricError("Logits and targets must have the same shape.")

        preds = binarize_predictions(logits, self.threshold)
        correct = (preds == targets).sum().item()
        total = targets.numel()
        self.correct += correct
        self.total += total
        logger.debug(f"Accuracy update - Correct: {correct}, Total: {total}")

    def compute(self) -> float:
        """
        Computes the binary accuracy.

        Returns:
            float: The accuracy value.
        """
        if self.total == 0:
            logger.warning("Accuracy compute called with total=0.")
            return 0.0
        accuracy = self.correct / self.total
        logger.debug(f"Computed Accuracy: {accuracy}")
        return accuracy


class Precision(Metric):
    """
    Computes the binary precision.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initializes the Precision metric.

        Args:
            threshold (float, optional): Threshold for binarization. Defaults to 0.5.
        """
        self.threshold: float = threshold
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state of the Precision metric.
        """
        self.true_positive: int = 0
        self.false_positive: int = 0
        logger.debug("Precision metric state has been reset.")

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the Precision metric with new predictions and targets.

        Args:
            logits (torch.Tensor): Predicted logits (before sigmoid), shape (N, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (N, 1, H, W).

        Raises:
            MetricError: If logits and targets have mismatched shapes.
        """
        if logits.shape != targets.shape:
            logger.error(f"Shape mismatch: logits shape {logits.shape} vs targets shape {targets.shape}")
            raise MetricError("Logits and targets must have the same shape.")

        preds = binarize_predictions(logits, self.threshold)
        tp = ((preds == 1) & (targets == 1)).sum().item()
        fp = ((preds == 1) & (targets == 0)).sum().item()
        self.true_positive += tp
        self.false_positive += fp
        logger.debug(f"Precision update - TP: {tp}, FP: {fp}")

    def compute(self) -> float:
        """
        Computes the binary precision.

        Returns:
            float: The precision value.
        """
        denominator = self.true_positive + self.false_positive
        if denominator == 0:
            logger.warning("Precision compute called with denominator=0.")
            return 0.0
        precision = self.true_positive / denominator
        logger.debug(f"Computed Precision: {precision}")
        return precision


class Recall(Metric):
    """
    Computes the binary recall.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initializes the Recall metric.

        Args:
            threshold (float, optional): Threshold for binarization. Defaults to 0.5.
        """
        self.threshold: float = threshold
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state of the Recall metric.
        """
        self.true_positive: int = 0
        self.false_negative: int = 0
        logger.debug("Recall metric state has been reset.")

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the Recall metric with new predictions and targets.

        Args:
            logits (torch.Tensor): Predicted logits (before sigmoid), shape (N, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (N, 1, H, W).

        Raises:
            MetricError: If logits and targets have mismatched shapes.
        """
        if logits.shape != targets.shape:
            logger.error(f"Shape mismatch: logits shape {logits.shape} vs targets shape {targets.shape}")
            raise MetricError("Logits and targets must have the same shape.")

        preds = binarize_predictions(logits, self.threshold)
        tp = ((preds == 1) & (targets == 1)).sum().item()
        fn = ((preds == 0) & (targets == 1)).sum().item()
        self.true_positive += tp
        self.false_negative += fn
        logger.debug(f"Recall update - TP: {tp}, FN: {fn}")

    def compute(self) -> float:
        """
        Computes the binary recall.

        Returns:
            float: The recall value.
        """
        denominator = self.true_positive + self.false_negative
        if denominator == 0:
            logger.warning("Recall compute called with denominator=0.")
            return 0.0
        recall = self.true_positive / denominator
        logger.debug(f"Computed Recall: {recall}")
        return recall


class F1Score(Metric):
    """
    Computes the binary F1 Score.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initializes the F1Score metric.

        Args:
            threshold (float, optional): Threshold for binarization. Defaults to 0.5.
        """
        self.threshold: float = threshold
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state of the F1Score metric.
        """
        self.precision = Precision(self.threshold)
        self.recall = Recall(self.threshold)
        logger.debug("F1Score metric state has been reset.")

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the F1Score metric with new predictions and targets.

        Args:
            logits (torch.Tensor): Predicted logits (before sigmoid), shape (N, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (N, 1, H, W).

        Raises:
            MetricError: If logits and targets have mismatched shapes.
        """
        self.precision.update(logits, targets)
        self.recall.update(logits, targets)
        logger.debug("F1Score metric updated with new predictions and targets.")

    def compute(self) -> float:
        """
        Computes the binary F1 Score.

        Returns:
            float: The F1 Score value.
        """
        prec = self.precision.compute()
        rec = self.recall.compute()
        if prec + rec == 0:
            logger.warning("F1Score compute called with prec + rec = 0.")
            return 0.0
        f1 = 2 * (prec * rec) / (prec + rec)
        logger.debug(f"Computed F1 Score: {f1}")
        return f1


class IoU(Metric):
    """
    Computes the Intersection over Union (IoU) score.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initializes the IoU metric.

        Args:
            threshold (float, optional): Threshold for binarization. Defaults to 0.5.
        """
        self.threshold: float = threshold
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state of the IoU metric.
        """
        self.intersection: int = 0
        self.union: int = 0
        logger.debug("IoU metric state has been reset.")

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the IoU metric with new predictions and targets.

        Args:
            logits (torch.Tensor): Predicted logits (before sigmoid), shape (N, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (N, 1, H, W).

        Raises:
            MetricError: If logits and targets have mismatched shapes.
        """
        if logits.shape != targets.shape:
            logger.error(f"Shape mismatch: logits shape {logits.shape} vs targets shape {targets.shape}")
            raise MetricError("Logits and targets must have the same shape.")

        preds = binarize_predictions(logits, self.threshold)
        intersection = ((preds == 1) & (targets == 1)).sum().item()
        union = ((preds == 1) | (targets == 1)).sum().item()
        self.intersection += intersection
        self.union += union
        logger.debug(f"IoU update - Intersection: {intersection}, Union: {union}")

    def compute(self) -> float:
        """
        Computes the Intersection over Union (IoU) score.

        Returns:
            float: The IoU score.
        """
        if self.union == 0:
            logger.warning("IoU compute called with union=0.")
            return 0.0
        iou = self.intersection / self.union
        logger.debug(f"Computed IoU: {iou}")
        return iou


class DiceCoefficient(Metric):
    """
    Computes the Dice Coefficient.
    """

    def __init__(self, threshold: float = 0.5, smooth: float = 1.0):
        """
        Initializes the DiceCoefficient metric.

        Args:
            threshold (float, optional): Threshold for binarization. Defaults to 0.5.
            smooth (float, optional): Smoothing factor to prevent division by zero. Defaults to 1.0.
        """
        self.threshold: float = threshold
        self.smooth: float = smooth
        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state of the DiceCoefficient metric.
        """
        self.intersection: float = 0.0
        self.sum_preds: float = 0.0
        self.sum_targets: float = 0.0
        logger.debug("DiceCoefficient metric state has been reset.")

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the DiceCoefficient metric with new predictions and targets.

        Args:
            logits (torch.Tensor): Predicted logits (before sigmoid), shape (N, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (N, 1, H, W).

        Raises:
            MetricError: If logits and targets have mismatched shapes.
        """
        if logits.shape != targets.shape:
            logger.error(f"Shape mismatch: logits shape {logits.shape} vs targets shape {targets.shape}")
            raise MetricError("Logits and targets must have the same shape.")

        preds = binarize_predictions(logits, self.threshold)
        intersection = (preds * targets).sum().item()
        sum_preds = preds.sum().item()
        sum_targets = targets.sum().item()
        self.intersection += intersection
        self.sum_preds += sum_preds
        self.sum_targets += sum_targets
        logger.debug(f"DiceCoefficient update - Intersection: {intersection}, Sum_preds: {sum_preds}, Sum_targets: {sum_targets}")

    def compute(self) -> float:
        """
        Computes the Dice Coefficient.

        Returns:
            float: The Dice Coefficient.
        """
        denominator = self.sum_preds + self.sum_targets + self.smooth
        dice = (2 * self.intersection + self.smooth) / denominator if denominator != 0 else 0.0
        logger.debug(f"Computed Dice Coefficient: {dice}")
        return dice


def get_metrics(names: Optional[List[str]] = None, **kwargs) -> List[Metric]:
    """
    Factory function to retrieve a list of metric instances based on names.

    Args:
        names (List[str], optional): List of metric names to instantiate.
            Options include 'accuracy', 'precision', 'recall', 'f1', 'iou', 'dice'.
            If None, defaults to ['accuracy', 'precision', 'recall', 'f1', 'iou', 'dice'].
        **kwargs: Additional keyword arguments to pass to the metric constructors.

    Returns:
        List[Metric]: List of metric instances.

    Raises:
        MetricError: If an unsupported metric name is provided.
    """
    if names is None:
        names = ['accuracy', 'precision', 'recall', 'f1', 'iou', 'dice']

    metric_map: Dict[str, Metric] = {}
    try:
        metric_map = {
            'accuracy': Accuracy(**kwargs),
            'precision': Precision(**kwargs),
            'recall': Recall(**kwargs),
            'f1': F1Score(**kwargs),
            'iou': IoU(**kwargs),
            'dice': DiceCoefficient(**kwargs),
        }
        logger.info("Initialized all requested metrics.")
    except Exception as e:
        logger.error(f"Error initializing metrics: {e}")
        raise MetricError(f"Error initializing metrics: {e}") from e

    metrics = []
    for name in names:
        key = name.lower()
        if key in metric_map:
            metrics.append(metric_map[key])
            logger.info(f"Added metric: {key}")
        else:
            logger.error(f"Unsupported metric: {name}")
            raise MetricError(f"Unsupported metric: {name}")

    return metrics
