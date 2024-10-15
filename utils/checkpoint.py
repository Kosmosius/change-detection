# utils/checkpoint.py

import os
from pathlib import Path
import torch
import logging
from typing import Optional, Tuple, Any
from contextlib import contextmanager
import tempfile
import shutil

logger = logging.getLogger(__name__)


@contextmanager
def atomic_save(save_path: Path, *args, **kwargs):
    """
    Context manager to save files atomically. Writes to a temporary file first and then renames it.
    
    Args:
        save_path (Path): The final path where the file should be saved.
        *args: Arguments to pass to torch.save().
        **kwargs: Keyword arguments to pass to torch.save().
    
    Yields:
        None
    """
    temp_dir = save_path.parent
    with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as tmp_file:
        temp_path = Path(tmp_file.name)
    try:
        torch.save(*args, **kwargs, f=temp_path)
        shutil.move(str(temp_path), str(save_path))
        logger.debug(f"Atomic save successful: '{save_path}'")
    except Exception as e:
        logger.error(f"Atomic save failed for '{save_path}': {e}")
        if temp_path.exists():
            temp_path.unlink()
        raise


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: Union[str, Path],
    filename: str = 'checkpoint.pth',
    is_best: bool = False
) -> None:
    """
    Saves the training checkpoint atomically.
    
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch number.
        loss (float): Training loss at the current epoch.
        checkpoint_dir (str or Path): Directory to save the checkpoint.
        filename (str, optional): Name of the checkpoint file. Defaults to 'checkpoint.pth'.
        is_best (bool, optional): Flag indicating if this checkpoint has the best validation loss. Defaults to False.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / filename
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    try:
        atomic_save(checkpoint_path, checkpoint)
        logger.info(f"Checkpoint saved at '{checkpoint_path}' (Epoch: {epoch}, Loss: {loss:.4f})")
        
        if is_best:
            best_path = checkpoint_dir / 'best_model.pth'
            atomic_save(best_path, checkpoint)
            logger.info(f"Best checkpoint updated at '{best_path}'")
    except Exception as e:
        logger.error(f"Failed to save checkpoint at '{checkpoint_path}': {e}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: Union[str, Path],
    map_location: Optional[torch.device] = None
) -> Tuple[Optional[int], Optional[float]]:
    """
    Loads the training checkpoint.
    
    Args:
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into. Defaults to None.
        checkpoint_path (str or Path): Path to the checkpoint file.
        map_location (torch.device, optional): Device to map the checkpoint to. Defaults to None.
    
    Returns:
        Tuple[Optional[int], Optional[float]]: A tuple containing the loaded epoch and loss.
            Returns (None, None) if loading fails.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        logger.error(f"Checkpoint file '{checkpoint_path}' does not exist.")
        return None, None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model state loaded from '{checkpoint_path}'.")
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Optimizer state loaded from '{checkpoint_path}'.")
        
        epoch = checkpoint.get('epoch')
        loss = checkpoint.get('loss')
        logger.info(f"Checkpoint loaded from '{checkpoint_path}' (Epoch: {epoch}, Loss: {loss:.4f})")
        return epoch, loss
    except Exception as e:
        logger.error(f"Failed to load checkpoint from '{checkpoint_path}': {e}")
        return None, None


def save_model(
    model: torch.nn.Module,
    checkpoint_dir: Union[str, Path],
    filename: str = 'model.pth'
) -> None:
    """
    Saves only the model's state dictionary for inference atomically.
    
    Args:
        model (torch.nn.Module): The model to save.
        checkpoint_dir (str or Path): Directory to save the model.
        filename (str, optional): Name of the model file. Defaults to 'model.pth'.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / filename
    
    try:
        atomic_save(model_path, model.state_dict())
        logger.info(f"Model state dictionary saved at '{model_path}'")
    except Exception as e:
        logger.error(f"Failed to save model state dictionary at '{model_path}': {e}")


def load_model(
    model: torch.nn.Module,
    model_path: Union[str, Path],
    map_location: Optional[torch.device] = None
) -> bool:
    """
    Loads the model's state dictionary from a file.
    
    Args:
        model (torch.nn.Module): The model to load the state into.
        model_path (str or Path): Path to the model file.
        map_location (torch.device, optional): Device to map the model to. Defaults to None.
    
    Returns:
        bool: True if loading was successful, False otherwise.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        logger.error(f"Model file '{model_path}' does not exist.")
        return False
    
    try:
        state_dict = torch.load(model_path, map_location=map_location)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"Model loaded from '{model_path}' and set to evaluation mode.")
        return True
    except Exception as e:
        logger.error(f"Failed to load model from '{model_path}': {e}")
        return False
