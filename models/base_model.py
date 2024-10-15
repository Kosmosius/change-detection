# models/base_model.py

from abc import ABC, abstractmethod
import torch.nn as nn
import torch
import logging

logger = logging.getLogger(__name__)

class BaseModel(nn.Module, ABC):
    """
    Abstract Base Model class that all models should inherit from.
    Defines the interface and common functionalities.
    """

    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Model initialized on device: {self.device}")

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        Must be implemented by all subclasses.
        """
        pass

    def to_device(self):
        """
        Moves the model to the specified device.
        """
        self.to(self.device)
        logger.info(f"Model moved to device: {self.device}")

    def freeze_parameters(self):
        """
        Freezes all parameters of the model.
        Useful for transfer learning.
        """
        for param in self.parameters():
            param.requires_grad = False
        logger.info("All model parameters have been frozen.")

    def unfreeze_parameters(self):
        """
        Unfreezes all parameters of the model.
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

    def summary(self):
        """
        Prints a summary of the model architecture and parameter count.
        """
        from torchsummary import summary
        summary(self, input_size=self.get_input_size())

    @abstractmethod
    def get_input_size(self) -> tuple:
        """
        Returns the expected input size for the model.
        Must be implemented by all subclasses.
        
        Returns:
            tuple: Input size (channels, height, width).
        """
        pass
