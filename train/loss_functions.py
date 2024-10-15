# train/loss_functions.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LossFunctionError(Exception):
    """Custom exception for loss function related errors."""
    pass


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation tasks.
    """

    def __init__(self, smooth: float = 1.0):
        """
        Initializes the DiceLoss module.

        Args:
            smooth (float, optional): Smoothing factor to prevent division by zero. Defaults to 1.0.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Dice Loss between logits and targets.

        Args:
            logits (torch.Tensor): Predicted logits (before sigmoid), shape (N, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (N, 1, H, W).

        Returns:
            torch.Tensor: Dice Loss value.

        Raises:
            LossFunctionError: If logits and targets have mismatched shapes.
        """
        if logits.shape != targets.shape:
            logger.error(f"Shape mismatch: logits shape {logits.shape} vs targets shape {targets.shape}")
            raise LossFunctionError("Logits and targets must have the same shape.")

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        loss = 1 - dice

        logger.debug(f"Dice Loss Computation: Intersection={intersection.item()}, Dice={dice.item()}, Loss={loss.item()}")
        return loss


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss for binary segmentation tasks.
    """

    def __init__(self, smooth: float = 1.0):
        """
        Initializes the IoULoss module.

        Args:
            smooth (float, optional): Smoothing factor to prevent division by zero. Defaults to 1.0.
        """
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the IoU Loss between logits and targets.

        Args:
            logits (torch.Tensor): Predicted logits (before sigmoid), shape (N, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (N, 1, H, W).

        Returns:
            torch.Tensor: IoU Loss value.

        Raises:
            LossFunctionError: If logits and targets have mismatched shapes.
        """
        if logits.shape != targets.shape:
            logger.error(f"Shape mismatch: logits shape {logits.shape} vs targets shape {targets.shape}")
            raise LossFunctionError("Logits and targets must have the same shape.")

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        total = probs_flat.sum() + targets_flat.sum()
        union = total - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1 - iou

        logger.debug(f"IoU Loss Computation: Intersection={intersection.item()}, IoU={iou.item()}, Loss={loss.item()}")
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initializes the FocalLoss module.

        Args:
            alpha (float, optional): Weighting factor for the rare class. Defaults to 0.25.
            gamma (float, optional): Focusing parameter for modulating factor (1-pt)^gamma. Defaults to 2.0.
            reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Defaults to 'mean'.

        Raises:
            ValueError: If an unsupported reduction method is provided.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        if reduction not in ['none', 'mean', 'sum']:
            logger.error(f"Unsupported reduction method: {reduction}")
            raise ValueError(f"Unsupported reduction method: {reduction}")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Focal Loss between logits and targets.

        Args:
            logits (torch.Tensor): Predicted logits (before sigmoid), shape (N, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (N, 1, H, W).

        Returns:
            torch.Tensor: Focal Loss value.

        Raises:
            LossFunctionError: If logits and targets have mismatched shapes.
        """
        if logits.shape != targets.shape:
            logger.error(f"Shape mismatch: logits shape {logits.shape} vs targets shape {targets.shape}")
            raise LossFunctionError("Logits and targets must have the same shape.")

        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        targets = targets.type_as(probs)

        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * BCE_loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        # If reduction is 'none', return the unreduced loss tensor

        logger.debug(f"Focal Loss Computation: alpha={self.alpha}, gamma={self.gamma}, Reduction={self.reduction}, Loss={loss.item() if self.reduction != 'none' else 'Unreduced tensor'}")
        return loss


def get_loss_function(name: str, **kwargs) -> nn.Module:
    """
    Factory function to retrieve the desired loss function.

    Args:
        name (str): Name of the loss function ('bce', 'dice', 'iou', 'focal').
        **kwargs: Additional keyword arguments for the loss function.

    Returns:
        nn.Module: Instance of the specified loss function.

    Raises:
        ValueError: If an unsupported loss function name is provided.
    """
    name = name.lower()
    try:
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
            logger.error(f"Unsupported loss function: {name}")
            raise ValueError(f"Unsupported loss function: {name}")
    except Exception as e:
        logger.error(f"Error initializing loss function '{name}': {e}")
        raise LossFunctionError(f"Failed to initialize loss function '{name}'.") from e
