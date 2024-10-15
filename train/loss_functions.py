# train/loss_functions.py

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation tasks.
    """

    def __init__(self, smooth: float = 1.0):
        """
        Initialize the DiceLoss module.

        Parameters
        ----------
        smooth : float, optional
            Smoothing factor to prevent division by zero. Defaults to 1.0.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice Loss between logits and targets.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits (before sigmoid), shape (N, 1, H, W).
        targets : torch.Tensor
            Ground truth binary masks, shape (N, 1, H, W).

        Returns
        -------
        torch.Tensor
            Dice Loss value.

        Raises
        ------
        ValueError
            If logits and targets have mismatched shapes.
        """
        if logits.shape != targets.shape:
            logger.error("Shape mismatch: logits shape %s vs targets shape %s", logits.shape, targets.shape)
            raise ValueError("Logits and targets must have the same shape.")

        probs = torch.sigmoid(logits)
        intersection = torch.sum(probs * targets)
        sum_probs = torch.sum(probs)
        sum_targets = torch.sum(targets)

        dice = (2.0 * intersection + self.smooth) / (sum_probs + sum_targets + self.smooth)
        loss = 1.0 - dice

        return loss


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss for binary segmentation tasks.
    """

    def __init__(self, smooth: float = 1.0):
        """
        Initialize the IoULoss module.

        Parameters
        ----------
        smooth : float, optional
            Smoothing factor to prevent division by zero. Defaults to 1.0.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the IoU Loss between logits and targets.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits (before sigmoid), shape (N, 1, H, W).
        targets : torch.Tensor
            Ground truth binary masks, shape (N, 1, H, W).

        Returns
        -------
        torch.Tensor
            IoU Loss value.

        Raises
        ------
        ValueError
            If logits and targets have mismatched shapes.
        """
        if logits.shape != targets.shape:
            logger.error("Shape mismatch: logits shape %s vs targets shape %s", logits.shape, targets.shape)
            raise ValueError("Logits and targets must have the same shape.")

        probs = torch.sigmoid(logits)
        intersection = torch.sum(probs * targets)
        sum_probs = torch.sum(probs)
        sum_targets = torch.sum(targets)
        union = sum_probs + sum_targets - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - iou

        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize the FocalLoss module.

        Parameters
        ----------
        alpha : float, optional
            Weighting factor for the rare class. Defaults to 0.25.
        gamma : float, optional
            Focusing parameter for modulating factor (1 - pt)^gamma. Defaults to 2.0.
        reduction : str, optional
            Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'. Defaults to 'mean'.

        Raises
        ------
        ValueError
            If an unsupported reduction method is provided.
        """
        super().__init__()
        if reduction not in {'none', 'mean', 'sum'}:
            raise ValueError(f"Unsupported reduction method: {reduction}")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Focal Loss between logits and targets.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits (before sigmoid), shape (N, 1, H, W).
        targets : torch.Tensor
            Ground truth binary masks, shape (N, 1, H, W).

        Returns
        -------
        torch.Tensor
            Focal Loss value.

        Raises
        ------
        ValueError
            If logits and targets have mismatched shapes.
        """
        if logits.shape != targets.shape:
            logger.error("Shape mismatch: logits shape %s vs targets shape %s", logits.shape, targets.shape)
            raise ValueError("Logits and targets must have the same shape.")

        probs = torch.sigmoid(logits)
        targets = targets.type_as(probs)

        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * BCE_loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        # If reduction is 'none', return the unreduced loss tensor

        return loss


def get_loss_function(name: str, **kwargs) -> nn.Module:
    """
    Factory function to retrieve the desired loss function.

    Parameters
    ----------
    name : str
        Name of the loss function ('bce', 'dice', 'iou', 'focal').
    **kwargs
        Additional keyword arguments for the loss function.

    Returns
    -------
    nn.Module
        Instance of the specified loss function.

    Raises
    ------
    ValueError
        If an unsupported loss function name is provided.
    """
    name = name.lower()
    if name == 'bce':
        logger.info("Using Binary Cross Entropy with Logits Loss.")
        return nn.BCEWithLogitsLoss(**kwargs)
    elif name == 'dice':
        logger.info("Using Dice Loss.")
        return DiceLoss(**kwargs)
    elif name == 'iou':
        logger.info("Using IoU Loss.")
        return IoULoss(**kwargs)
    elif name == 'focal':
        logger.info("Using Focal Loss.")
        return FocalLoss(**kwargs)
    else:
        logger.error("Unsupported loss function: %s", name)
        raise ValueError(f"Unsupported loss function: {name}")
