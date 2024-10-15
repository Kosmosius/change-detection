# train/metrics.py

import torch
import torch.nn as nn
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

def binarize_predictions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Binarizes the predictions based on the specified threshold.
    
    Args:
        logits (torch.Tensor): Predicted logits (before sigmoid), shape (N, 1, H, W).
        threshold (float): Threshold for binarization. Defaults to 0.5.
    
    Returns:
        torch.Tensor: Binarized predictions, shape (N, 1, H, W).
    """
    probs = torch.sigmoid(logits)
    return (probs >= threshold).float()

class Metric:
    """
    Base class for all metrics.
    """
    def reset(self):
        raise NotImplementedError
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        raise NotImplementedError
    
    def compute(self) -> float:
        raise NotImplementedError

class Accuracy(Metric):
    """
    Computes the binary accuracy.
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.correct = 0
        self.total = 0
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = binarize_predictions(logits, self.threshold)
        self.correct += (preds == targets).sum().item()
        self.total += targets.numel()
    
    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

class Precision(Metric):
    """
    Computes the binary precision.
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.true_positive = 0
        self.false_positive = 0
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = binarize_predictions(logits, self.threshold)
        self.true_positive += ((preds == 1) & (targets == 1)).sum().item()
        self.false_positive += ((preds == 1) & (targets == 0)).sum().item()
    
    def compute(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator > 0 else 0.0

class Recall(Metric):
    """
    Computes the binary recall.
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.true_positive = 0
        self.false_negative = 0
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = binarize_predictions(logits, self.threshold)
        self.true_positive += ((preds == 1) & (targets == 1)).sum().item()
        self.false_negative += ((preds == 0) & (targets == 1)).sum().item()
    
    def compute(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator > 0 else 0.0

class F1Score(Metric):
    """
    Computes the binary F1 Score.
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.precision = Precision(self.threshold)
        self.recall = Recall(self.threshold)
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        self.precision.update(logits, targets)
        self.recall.update(logits, targets)
    
    def compute(self) -> float:
        prec = self.precision.compute()
        rec = self.recall.compute()
        if prec + rec == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)

class IoU(Metric):
    """
    Computes the Intersection over Union (IoU) score.
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.intersection = 0
        self.union = 0
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = binarize_predictions(logits, self.threshold)
        self.intersection += ((preds == 1) & (targets == 1)).sum().item()
        self.union += ((preds == 1) | (targets == 1)).sum().item()
    
    def compute(self) -> float:
        return self.intersection / self.union if self.union > 0 else 0.0

class DiceCoefficient(Metric):
    """
    Computes the Dice Coefficient.
    """
    def __init__(self, threshold: float = 0.5, smooth: float = 1.0):
        self.threshold = threshold
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        self.intersection = 0
        self.sum_preds = 0
        self.sum_targets = 0
    
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = binarize_predictions(logits, self.threshold)
        self.intersection += (preds * targets).sum().item()
        self.sum_preds += preds.sum().item()
        self.sum_targets += targets.sum().item()
    
    def compute(self) -> float:
        dice = (2 * self.intersection + self.smooth) / (self.sum_preds + self.sum_targets + self.smooth)
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
        ValueError: If an unsupported metric name is provided.
    """
    if names is None:
        names = ['accuracy', 'precision', 'recall', 'f1', 'iou', 'dice']
    
    metric_map: Dict[str, Metric] = {
        'accuracy': Accuracy(**kwargs),
        'precision': Precision(**kwargs),
        'recall': Recall(**kwargs),
        'f1': F1Score(**kwargs),
        'iou': IoU(**kwargs),
        'dice': DiceCoefficient(**kwargs),
    }
    
    metrics = []
    for name in names:
        key = name.lower()
        if key in metric_map:
            metrics.append(metric_map[key])
            logger.info(f"Added metric: {key}")
        else:
            logger.error(f"Unsupported metric: {name}")
            raise ValueError(f"Unsupported metric: {name}")
    
    return metrics
