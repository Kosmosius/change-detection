# utils/checkpoint.py

import os
from pathlib import Path
import torch
import logging
from typing import Optional, Tuple, Any, Union
import tempfile

logger = logging.getLogger(__name__)


def atomic_save(save_path: Path, data: Any) -> None:
    """
    Saves data to a file atomically.

    Writes the data to a temporary file in the same directory and then
    moves it to the target path. This ensures that the file is not corrupted
    if the saving process is interrupted.

    Args:
        save_path (Path): The final path where the file should be saved.
        data (Any): Data to save using torch.save().

    Raises:
        OSError: If the file cannot be saved due to an OS error.
        IOError: If an I/O operation fails.
    """
    temp_dir = save_path.parent
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as tmp_file:
            temp_path = Path(tmp_file.name)
            torch.save(data, tmp_file)
        temp_path.replace(save_path)
        logger.debug("Atomic save successful: '%s'", save_path)
    except (OSError, IOError) as e:
        logger.error("Atomic save failed for '%s': %s", save_path, e)
        if temp_path and temp_path.exists():
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

    Creates a checkpoint containing the model state, optimizer state,
    current epoch, and loss. Saves the checkpoint to the specified directory.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): Current epoch number.
        loss (float): Training loss at the current epoch.
        checkpoint_dir (Union[str, Path]): Directory to save the checkpoint.
        filename (str, optional): Name of the checkpoint file. Defaults to 'checkpoint.pth'.
        is_best (bool, optional): If True, also saves a copy as 'best_model.pth'.

    Raises:
        OSError: If saving the checkpoint fails due to an OS error.
        IOError: If an I/O operation fails.
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
        logger.info("Checkpoint saved at '%s' (Epoch: %d, Loss: %.4f)", checkpoint_path, epoch, loss)

        if is_best:
            best_path = checkpoint_dir / 'best_model.pth'
            atomic_save(best_path, checkpoint)
            logger.info("Best checkpoint updated at '%s'", best_path)
    except (OSError, IOError) as e:
        logger.error("Failed to save checkpoint at '%s': %s", checkpoint_path, e)
        raise


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None
) -> Tuple[int, float]:
    """
    Loads the training checkpoint.

    Restores the model and optimizer state from a checkpoint file.
    Returns the epoch and loss stored in the checkpoint.

    Args:
        model (torch.nn.Module): The model to load the state into.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer to load the state into.
        checkpoint_path (Union[str, Path]): Path to the checkpoint file.
        map_location (Optional[Union[str, torch.device]]): Device to map the checkpoint to.

    Returns:
        Tuple[int, float]: A tuple containing the loaded epoch and loss.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If the checkpoint fails to load.
        KeyError: If expected keys are missing in the checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        logger.error("Checkpoint file '%s' does not exist.", checkpoint_path)
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' does not exist.")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model state loaded from '%s'.", checkpoint_path)

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state loaded from '%s'.", checkpoint_path)

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logger.info("Checkpoint loaded from '%s' (Epoch: %d, Loss: %.4f)", checkpoint_path, epoch, loss)
        return epoch, loss
    except (OSError, IOError, RuntimeError, KeyError) as e:
        logger.error("Failed to load checkpoint from '%s': %s", checkpoint_path, e)
        raise


def save_model(
    model: torch.nn.Module,
    checkpoint_dir: Union[str, Path],
    filename: str = 'model.pth'
) -> None:
    """
    Saves only the model's state dictionary for inference atomically.

    Args:
        model (torch.nn.Module): The model to save.
        checkpoint_dir (Union[str, Path]): Directory to save the model.
        filename (str, optional): Name of the model file. Defaults to 'model.pth'.

    Raises:
        OSError: If saving the model fails due to an OS error.
        IOError: If an I/O operation fails.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / filename

    try:
        atomic_save(model_path, model.state_dict())
        logger.info("Model state dictionary saved at '%s'", model_path)
    except (OSError, IOError) as e:
        logger.error("Failed to save model state dictionary at '%s': %s", model_path, e)
        raise


def load_model(
    model: torch.nn.Module,
    model_path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None
) -> None:
    """
    Loads the model's state dictionary from a file.

    Sets the model to evaluation mode after loading.

    Args:
        model (torch.nn.Module): The model to load the state into.
        model_path (Union[str, Path]): Path to the model file.
        map_location (Optional[Union[str, torch.device]]): Device to map the model to.

    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If loading the model fails.
        KeyError: If expected keys are missing in the state dictionary.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        logger.error("Model file '%s' does not exist.", model_path)
        raise FileNotFoundError(f"Model file '{model_path}' does not exist.")

    try:
        state_dict = torch.load(model_path, map_location=map_location)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info("Model loaded from '%s' and set to evaluation mode.", model_path)
    except (OSError, IOError, RuntimeError, KeyError) as e:
        logger.error("Failed to load model from '%s': %s", model_path, e)
        raise
