# models/base_model.py

from abc import ABC, abstractmethod
import torch.nn as nn
import torch
import logging
from typing import Tuple, Optional
from torchsummary import summary

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
            device (torch.device, optional): The device to run the model on.
                If None, automatically selects CUDA if available, else CPU.
        """
        super(BaseModel, self).__init__()
        self.device: torch.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Model initialized on device: {self.device}")

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.
        Must be implemented by all subclasses.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        
        Returns:
            torch.Tensor: The output tensor of the model.
        """
        pass

    def to_device(self, device: Optional[torch.device] = None) -> None:
        """
        Moves the model to the specified device.
        
        Args:
            device (torch.device, optional): The device to move the model to.
                If None, uses the device attribute of the model.
        """
        target_device = device if device else self.device
        self.to(target_device)
        logger.info(f"Model moved to device: {target_device}")

    def freeze_parameters(self) -> None:
        """
        Freezes all parameters of the model.
        Useful for transfer learning where only certain layers are trained.
        """
        for param in self.parameters():
            param.requires_grad = False
        logger.info("All model parameters have been frozen.")

    def unfreeze_parameters(self) -> None:
        """
        Unfreezes all parameters of the model.
        Allows all layers to be trained.
        """
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All model parameters have been unfrozen.")

    def count_parameters(self) -> int:
        """
        Counts the number of trainable parameters in the model.
        
        Returns:
            int: Number of trainable parameters.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {total_params}")
        return total_params

    def summary(self) -> None:
        """
        Prints a summary of the model architecture and parameter count.
        Requires that `get_input_size` is implemented by the subclass.
        
        Raises:
            NotImplementedError: If `get_input_size` is not implemented.
            ValueError: If `torchsummary` is not installed.
        """
        try:
            input_size = self.get_input_size()
        except NotImplementedError as e:
            logger.error(f"Method get_input_size not implemented: {e}")
            raise

        try:
            summary(self, input_size=input_size, device=str(self.device))
        except Exception as e:
            logger.error(f"Failed to generate model summary: {e}")
            raise ValueError("Could not generate model summary. Ensure that torchsummary supports your model architecture.") from e

    @abstractmethod
    def get_input_size(self) -> Tuple[int, int, int]:
        """
        Returns the expected input size for the model.
        Must be implemented by all subclasses.
        
        Returns:
            Tuple[int, int, int]: Input size in the format (channels, height, width).
        """
        pass
