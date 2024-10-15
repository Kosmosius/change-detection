# models/siamese_unet.py

import torch
import torch.nn as nn
import logging
from typing import List, Tuple, Optional, Union

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


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
            feature_maps (List[int], optional): List defining the number of feature maps at each layer.
                Defaults to [64, 128, 256, 512].
            device (torch.device, optional): Device to run the model on.
                If None, uses the default device from BaseModel.
        """
        super(SiameseUNet, self).__init__(device=device)
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.feature_maps: List[int] = feature_maps if feature_maps else [64, 128, 256, 512]

        # Shared Encoder
        self.encoder: nn.Sequential = self._build_encoder()

        # Shared Decoder
        self.decoder: nn.Module = self._build_decoder()

        # Final Convolution
        self.final_conv: nn.Conv2d = nn.Conv2d(self.feature_maps[0], self.out_channels, kernel_size=1)

        logger.info("Initialized SiameseUNet architecture.")

    def _build_encoder(self) -> nn.Sequential:
        """
        Builds the shared encoder part of the Siamese U-Net.

        Returns:
            nn.Sequential: Encoder network.
        """
        try:
            layers: List[nn.Module] = []
            prev_channels: int = self.in_channels
            for idx, fm in enumerate(self.feature_maps):
                layers.append(nn.Conv2d(prev_channels, fm, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(fm))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                logger.debug(f"Encoder Layer {idx + 1}: Conv2d({prev_channels}, {fm}, 3, 1), "
                             f"BatchNorm2d({fm}), ReLU, MaxPool2d(2)")
                prev_channels = fm
            encoder = nn.Sequential(*layers)
            logger.info("Shared encoder built successfully.")
            return encoder
        except Exception as e:
            logger.error(f"Failed to build encoder: {e}")
            raise ValueError("Invalid encoder configuration.") from e

    def _build_decoder(self) -> nn.Module:
        """
        Builds the shared decoder part of the Siamese U-Net.

        Returns:
            nn.Sequential: Decoder network.
        """
        try:
            layers: List[nn.Module] = []
            reversed_features: List[int] = list(reversed(self.feature_maps[:-1]))
            prev_channels: int = self.feature_maps[-1]
            for idx, fm in enumerate(reversed_features):
                layers.append(nn.ConvTranspose2d(prev_channels, fm, kernel_size=2, stride=2))
                layers.append(nn.BatchNorm2d(fm))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Conv2d(fm * 2, fm, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(fm))
                layers.append(nn.ReLU(inplace=True))
                logger.debug(f"Decoder Layer {idx + 1}: ConvTranspose2d({prev_channels}, {fm}, 2, 2), "
                             f"BatchNorm2d({fm}), ReLU, Conv2d({fm * 2}, {fm}, 3, 1), "
                             f"BatchNorm2d({fm}), ReLU")
                prev_channels = fm
            decoder = nn.Sequential(*layers)
            logger.info("Shared decoder built successfully.")
            return decoder
        except Exception as e:
            logger.error(f"Failed to build decoder: {e}")
            raise ValueError("Invalid decoder configuration.") from e

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Siamese U-Net model with concatenated encoder outputs.

        Args:
            img1 (torch.Tensor): First input image tensor of shape (batch_size, channels, height, width).
            img2 (torch.Tensor): Second input image tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output change map tensor of shape (batch_size, out_channels, height, width).
        """
        try:
            # Move inputs to the correct device
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            logger.debug("Input images moved to device.")

            # Pass both images through the shared encoder
            enc1 = self.encoder(img1)
            enc2 = self.encoder(img2)
            logger.debug("Images passed through the encoder.")

            # Concatenate encoder outputs along the channel dimension
            enc = torch.cat([enc1, enc2], dim=1)  # Shape: (batch_size, 2 * feature_maps[-1], H, W)
            logger.debug(f"Encoder outputs concatenated. Shape: {enc.shape}")

            # Pass through decoder
            dec = self.decoder(enc)
            logger.debug(f"Decoder output shape: {dec.shape}")

            # Final convolution
            out = self.final_conv(dec)
            logger.debug(f"Final convolution output shape: {out.shape}")

            # Apply sigmoid activation for binary change detection
            out = torch.sigmoid(out)
            logger.debug("Sigmoid activation applied to output.")

            return out
        except Exception as e:
            logger.error(f"Error during forward pass: {e}")
            raise

    def get_input_size(self) -> Tuple[int, int, int]:
        """
        Specifies the input size for each image.

        Returns:
            Tuple[int, int, int]: Input size in the format (channels, height, width).
        """
        return (self.in_channels, 256, 256)  # Example input size

    def freeze_parameters(self) -> None:
        """
        Freezes the encoder parameters to prevent them from being updated during training.
        """
        super().freeze_parameters()
        logger.info("Encoder parameters have been frozen.")

    def unfreeze_parameters(self) -> None:
        """
        Unfreezes the encoder parameters to allow them to be updated during training.
        """
        super().unfreeze_parameters()
        logger.info("Encoder parameters have been unfrozen.")
