# train/loss_functions.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation tasks.
    """
    def __init__(self, smooth: float = 1.0):
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
        """
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        loss = 1 - dice
        return loss

class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss for binary segmentation tasks.
    """
    def __init__(self, smooth: float = 1.0):
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
        """
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        total = probs.sum() + targets.sum()
        union = total - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1 - iou
        return loss

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Focal Loss between logits and targets.
        
        Args:
            logits (torch.Tensor): Predicted logits (before sigmoid), shape (N, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (N, 1, H, W).
        
        Returns:
            torch.Tensor: Focal Loss value.
        """
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        targets = targets.type_as(probs)
        
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
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
