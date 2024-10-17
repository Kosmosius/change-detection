# models/siamese_unet.py

import logging
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """
    A module consisting of two consecutive convolutional layers, each followed by
    batch normalization (optional) and a ReLU activation.
    """

    def __init__(self, in_channels: int, out_channels: int, use_batch_norm: bool = True):
        """
        Initializes the DoubleConv module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_batch_norm (bool): Whether to use batch normalization. Defaults to True.
        """
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        if use_batch_norm:
            layers.insert(1, nn.BatchNorm2d(out_channels))
            layers.insert(4, nn.BatchNorm2d(out_channels))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """
    A module for downsampling that applies max pooling followed by DoubleConv.
    """

    def __init__(self, in_channels: int, out_channels: int, use_batch_norm: bool = True):
        """
        Initializes the Down module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_batch_norm (bool): Whether to use batch normalization.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_batch_norm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    A module for upsampling that applies transposed convolution (upscaling)
    followed by DoubleConv.
    """

    def __init__(
        self, in_channels: int, out_channels: int, use_batch_norm: bool = True, bilinear: bool = True
    ):
        """
        Initializes the Up module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_batch_norm (bool): Whether to use batch normalization.
            bilinear (bool): Whether to use bilinear upsampling. If False, uses transposed convolution.
        """
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, use_batch_norm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_batch_norm)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Adjust shape if necessary due to padding differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY != 0 or diffX != 0:
            x1 = F.pad(
                x1,
                [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
            )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SiameseUNet(BaseModel):
    """
    Siamese U-Net architecture for change detection tasks.
    Processes two input images through shared encoders and decoders with skip connections.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        feature_maps: Optional[List[int]] = None,
        use_batch_norm: bool = True,
        bilinear: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initializes the SiameseUNet model.

        Args:
            in_channels (int): Number of input channels for each image.
            out_channels (int): Number of output channels.
            feature_maps (Optional[List[int]]): Number of feature maps at each layer.
            use_batch_norm (bool): Whether to use batch normalization.
            bilinear (bool): Whether to use bilinear upsampling in the decoder.
            device (Optional[torch.device]): Device to run the model on.
        """
        super().__init__(device=device)

        if feature_maps is None:
            feature_maps = [64, 128, 256, 512]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.use_batch_norm = use_batch_norm
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(self.in_channels, self.feature_maps[0], use_batch_norm)
        self.down1 = Down(self.feature_maps[0], self.feature_maps[1], use_batch_norm)
        self.down2 = Down(self.feature_maps[1], self.feature_maps[2], use_batch_norm)
        self.down3 = Down(self.feature_maps[2], self.feature_maps[3], use_batch_norm)

        # Bottleneck
        self.bottleneck = DoubleConv(self.feature_maps[3], self.feature_maps[3], use_batch_norm)

        # Decoder
        self.up1 = Up(
            self.feature_maps[3] * 2, self.feature_maps[2], use_batch_norm, bilinear
        )
        self.up2 = Up(
            self.feature_maps[2] * 2, self.feature_maps[1], use_batch_norm, bilinear
        )
        self.up3 = Up(
            self.feature_maps[1] * 2, self.feature_maps[0], use_batch_norm, bilinear
        )

        # Final Convolution
        self.outc = nn.Conv2d(self.feature_maps[0], self.out_channels, kernel_size=1)

        logger.info("Initialized SiameseUNet architecture.")

    def forward(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Siamese U-Net model with skip connections.

        Args:
            img1 (torch.Tensor): First input image tensor of shape (N, C, H, W).
            img2 (torch.Tensor): Second input image tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output change map tensor of shape (N, out_channels, H, W).
        """
        if img1.shape != img2.shape:
            raise ValueError("Input tensors img1 and img2 must have the same shape.")
        if img1.dim() != 4:
            raise ValueError("Input tensors must be 4-dimensional (N, C, H, W).")

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
        # The model can handle any input size that is divisible by 16 due to the pooling layers
        return (self.in_channels, 256, 256)

    def freeze_encoder(self) -> None:
        """
        Freezes the encoder parameters to prevent them from being updated during training.
        """
        for module in [self.inc, self.down1, self.down2, self.down3]:
            for param in module.parameters():
                param.requires_grad = False
        logger.debug("Encoder parameters have been frozen.")

    def unfreeze_encoder(self) -> None:
        """
        Unfreezes the encoder parameters to allow them to be updated during training.
        """
        for module in [self.inc, self.down1, self.down2, self.down3]:
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

    def summary(self, input_size: Optional[Tuple[int, int, int]] = None) -> None:
        """
        Prints a summary of the model architecture and parameter count.

        Args:
            input_size (Optional[Tuple[int, int, int]]): Input size in the format (channels, height, width).
                If None, uses the default input size from get_input_size().
        """
        from torchinfo import summary

        if input_size is None:
            input_size = self.get_input_size()

        try:
            summary(
                self,
                input_size=[(1, *input_size), (1, *input_size)],
                device=str(self.device),
            )
        except Exception as e:
            logger.error("Failed to generate model summary: %s", e)
            raise
