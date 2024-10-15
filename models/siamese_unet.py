# models/siamese_unet.py

import logging
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Optional: remove if not using batch normalization
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Optional
            nn.ReLU(inplace=True),
        ]
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Adjust shape if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SiameseUNet(BaseModel):
    """
    Siamese U-Net architecture for Change Detection tasks.
    Processes two input images through shared encoders and decoders with skip connections.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        feature_maps: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initializes the SiameseUNet model.

        Args:
            in_channels (int): Number of input channels for each image.
                Defaults to 3 (RGB images).
            out_channels (int): Number of output channels (e.g., 1 for binary change detection).
                Defaults to 1.
            feature_maps (Optional[List[int]]): List defining the number of feature maps at each layer.
                Defaults to [64, 128, 256, 512].
            device (Optional[torch.device]): Device to run the model on.
                If None, uses the default device from BaseModel.

        Raises:
            ValueError: If input parameters are invalid.
        """
        super().__init__(device=device)

        if in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")
        if out_channels <= 0:
            raise ValueError("out_channels must be a positive integer.")
        if feature_maps is None:
            feature_maps = [64, 128, 256, 512]
        elif not all(isinstance(fm, int) and fm > 0 for fm in feature_maps):
            raise ValueError("feature_maps must be a list of positive integers.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps

        # Build the encoder and decoder
        self.inc = DoubleConv(self.in_channels, self.feature_maps[0])
        self.down1 = Down(self.feature_maps[0], self.feature_maps[1])
        self.down2 = Down(self.feature_maps[1], self.feature_maps[2])
        self.down3 = Down(self.feature_maps[2], self.feature_maps[3])

        # Bottleneck
        self.bottleneck = DoubleConv(self.feature_maps[3], self.feature_maps[3])

        # Upsampling
        self.up1 = Up(self.feature_maps[3] * 2, self.feature_maps[2])
        self.up2 = Up(self.feature_maps[2] * 2, self.feature_maps[1])
        self.up3 = Up(self.feature_maps[1] * 2, self.feature_maps[0])

        self.outc = nn.Conv2d(self.feature_maps[0], self.out_channels, kernel_size=1)

        logger.info("Initialized SiameseUNet architecture.")

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Siamese U-Net model with skip connections.

        Args:
            img1 (torch.Tensor): First input image tensor of shape (batch_size, channels, height, width).
            img2 (torch.Tensor): Second input image tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output change map tensor of shape (batch_size, out_channels, height, width).

        Raises:
            ValueError: If input tensors have incompatible shapes.
        """
        if img1.shape != img2.shape:
            raise ValueError("Input tensors img1 and img2 must have the same shape.")
        if img1.dim() != 4:
            raise ValueError("Input tensors must be 4-dimensional (batch_size, channels, height, width).")

        # Encoder path for img1
        x1 = self.inc(img1)
        x1_down1 = self.down1(x1)
        x1_down2 = self.down2(x1_down1)
        x1_down3 = self.down3(x1_down2)
        x1_bottleneck = self.bottleneck(x1_down3)

        # Encoder path for img2
        y1 = self.inc(img2)
        y1_down1 = self.down1(y1)
        y1_down2 = self.down2(y1_down1)
        y1_down3 = self.down3(y1_down2)
        y1_bottleneck = self.bottleneck(y1_down3)

        # Compute the difference at the bottleneck
        diff_bottleneck = torch.abs(x1_bottleneck - y1_bottleneck)

        # Decoder path with skip connections
        x = self.up1(diff_bottleneck, torch.abs(x1_down3 - y1_down3))
        x = self.up2(x, torch.abs(x1_down2 - y1_down2))
        x = self.up3(x, torch.abs(x1_down1 - y1_down1))
        logits = self.outc(x)

        return logits

    def get_input_size(self) -> Tuple[int, int, int]:
        """
        Returns the expected input size for the model.

        Returns:
            Tuple[int, int, int]: Input size in the format (channels, height, width).
        """
        return (self.in_channels, 256, 256)  # Example input size

    def freeze_encoder(self) -> None:
        """
        Freezes the encoder parameters to prevent them from being updated during training.
        """
        for module in [self.inc, self.down1, self.down2, self.down3, self.bottleneck]:
            for param in module.parameters():
                param.requires_grad = False
        logger.debug("Encoder parameters have been frozen.")

    def unfreeze_encoder(self) -> None:
        """
        Unfreezes the encoder parameters to allow them to be updated during training.
        """
        for module in [self.inc, self.down1, self.down2, self.down3, self.bottleneck]:
            for param in module.parameters():
                param.requires_grad = True
        logger.debug("Encoder parameters have been unfrozen.")

    def count_parameters(self) -> int:
        """
        Counts the number of trainable parameters in the model.

        Returns:
            int: Number of trainable parameters.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Total trainable parameters: %d", total_params)
        return total_params

    def summary(self) -> None:
        """
        Prints a summary of the model architecture and parameter count.
        """
        from torchinfo import summary
        input_size = self.get_input_size()
        try:
            summary(self, input_size=[(1, *input_size), (1, *input_size)], device=str(self.device))
        except Exception as e:
            logger.error("Failed to generate model summary: %s", e)
            raise
