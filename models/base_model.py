# models/base_model.py

from abc import ABC, abstractmethod
import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torchinfo import summary  # Replaced torchsummary with torchinfo

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, ABC):
    """
    Abstract Base Model class that all models should inherit from.
    Defines the interface and common functionalities.

    Attributes:
        device (torch.device): The device on which the model is running.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initializes the BaseModel with device configuration.

        Args:
            device (Optional[torch.device]): The device to run the model on.
                If None, automatically selects CUDA if available, else CPU.
        """
        super().__init__()
        self.device: torch.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Model initialized on device: %s", self.device)

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.
        Must be implemented by all subclasses.

        Returns:
            torch.Tensor: The output tensor of the model.
        """
        pass

    def to_device(self, device: Optional[torch.device] = None) -> None:
        """
        Moves the model to the specified device.

        Args:
            device (Optional[torch.device]): The device to move the model to.
                If None, uses the device attribute of the model.
        """
        target_device = device or self.device
        self.to(target_device)
        logger.info("Model moved to device: %s", target_device)

    def freeze_parameters(self) -> None:
        """
        Freezes all parameters of the model.
        Useful for transfer learning where only certain layers are trained.
        """
        for param in self.parameters():
            param.requires_grad = False
        logger.debug("All model parameters have been frozen.")

    def unfreeze_parameters(self) -> None:
        """
        Unfreezes all parameters of the model.
        Allows all layers to be trained.
        """
        for param in self.parameters():
            param.requires_grad = True
        logger.debug("All model parameters have been unfrozen.")

    def count_parameters(self) -> int:
        """
        Counts the number of trainable parameters in the model.

        Returns:
            int: Number of trainable parameters.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Total trainable parameters: %d", total_params)
        return total_params

    def summary(self, input_size: Optional[Tuple[int, ...]] = None, batch_size: int = 1) -> None:
        """
        Prints a summary of the model architecture and parameter count.
        Requires that `get_input_size` is implemented by the subclass, or `input_size` is provided.

        Args:
            input_size (Optional[Tuple[int, ...]]): Input size in the format (channels, height, width).
                If None, calls `get_input_size()` to get the input size.
            batch_size (int): Batch size for the summary. Defaults to 1.

        Raises:
            NotImplementedError: If `get_input_size` is not implemented and `input_size` is not provided.
        """
        if input_size is None:
            input_size = self.get_input_size()
        input_shape = (batch_size, *input_size)
        summary(self, input_size=input_shape, device=str(self.device))

    @abstractmethod
    def get_input_size(self) -> Tuple[int, ...]:
        """
        Returns the expected input size for the model.
        Must be implemented by all subclasses.

        Returns:
            Tuple[int, ...]: Input size in the format (channels, height, width).
        """
        pass
