# utils/checkpoint.py

import os
from pathlib import Path
import torch
import logging

logger = logging.getLogger(__name__)

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, filename='checkpoint.pth'):
    """
    Saves the training checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch number.
        loss (float): Training loss at the current epoch.
        checkpoint_dir (str or Path): Directory to save the checkpoint.
        filename (str): Name of the checkpoint file.
    """
    checkpoint_path = Path(checkpoint_dir) / filename
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    try:
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved at '{checkpoint_path}'")
    except Exception as e:
        logger.error(f"Failed to save checkpoint at '{checkpoint_path}': {e}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Loads the training checkpoint.

    Args:
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        checkpoint_path (str or Path): Path to the checkpoint file.

    Returns:
        tuple: (epoch (int), loss (float)) if successful, otherwise (None, None).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        logger.error(f"Checkpoint file '{checkpoint_path}' does not exist.")
        return None, None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', None)
        loss = checkpoint.get('loss', None)
        logger.info(f"Checkpoint loaded from '{checkpoint_path}' (Epoch: {epoch}, Loss: {loss})")
        return epoch, loss
    except Exception as e:
        logger.error(f"Failed to load checkpoint from '{checkpoint_path}': {e}")
        return None, None

def save_model(model, checkpoint_dir, filename='model.pth'):
    """
    Saves only the model's state dictionary for inference.

    Args:
        model (torch.nn.Module): The model to save.
        checkpoint_dir (str or Path): Directory to save the model.
        filename (str): Name of the model file.
    """
    model_path = Path(checkpoint_dir) / filename
    try:
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved at '{model_path}'")
    except Exception as e:
        logger.error(f"Failed to save model at '{model_path}': {e}")

def load_model(model, model_path):
    """
    Loads the model's state dictionary from a file.

    Args:
        model (torch.nn.Module): The model to load the state into.
        model_path (str or Path): Path to the model file.

    Returns:
        bool: True if loading was successful, False otherwise.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        logger.error(f"Model file '{model_path}' does not exist.")
        return False

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        logger.info(f"Model loaded from '{model_path}' and set to evaluation mode.")
        return True
    except Exception as e:
        logger.error(f"Failed to load model from '{model_path}': {e}")
        return False
