# models/siamese_unet.py

import torch
import torch.nn as nn
import logging
from models.base_model import BaseModel

logger = logging.getLogger(__name__)

class SiameseUNet(BaseModel):
    """
    Siamese U-Net architecture for Change Detection tasks.
    Processes two input images through shared encoders and decoders with skip connections.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, feature_maps: list = [64, 128, 256, 512]):
        """
        Initializes the SiameseUNet model.
        
        Args:
            in_channels (int): Number of input channels for each image.
            out_channels (int): Number of output channels (e.g., 1 for binary change detection).
            feature_maps (list): List defining the number of feature maps at each layer.
        """
        super(SiameseUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps

        # Shared Encoder
        self.encoder = self.build_encoder()

        # Shared Decoder
        self.decoder = self.build_decoder()

        # Final Convolution
        self.final_conv = nn.Conv2d(feature_maps[0], out_channels, kernel_size=1)

        logger.info("Initialized SiameseUNet architecture.")

    def build_encoder(self) -> nn.Sequential:
        """
        Builds the shared encoder part of the Siamese U-Net.
        
        Returns:
            nn.Sequential: Encoder network.
        """
        layers = []
        prev_channels = self.in_channels
        for fm in self.feature_maps:
            layers.append(nn.Conv2d(prev_channels, fm, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(fm))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = fm
        encoder = nn.Sequential(*layers)
        return encoder

    def build_decoder(self) -> nn.Sequential:
        """
        Builds the shared decoder part of the Siamese U-Net.
        
        Returns:
            nn.Sequential: Decoder network.
        """
        layers = []
        reversed_features = list(reversed(self.feature_maps[:-1]))
        prev_channels = self.feature_maps[-1]
        for fm in reversed_features:
            layers.append(nn.ConvTranspose2d(prev_channels, fm, kernel_size=2, stride=2))
            layers.append(nn.BatchNorm2d(fm))
            layers.append(nn.ReLU(inplace=True))
            # Assuming skip connections concatenated from encoder, hence input channels doubled
            layers.append(nn.Conv2d(fm * 2, fm, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(fm))
            layers.append(nn.ReLU(inplace=True))
            prev_channels = fm
        decoder = nn.Sequential(*layers)
        return decoder

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Siamese U-Net model with concatenated encoder outputs.
        
        Args:
            img1 (torch.Tensor): First input image tensor.
            img2 (torch.Tensor): Second input image tensor.
        
        Returns:
            torch.Tensor: Output change map.
        """
        # Pass both images through the shared encoder
        enc1 = self.encoder(img1)
        enc2 = self.encoder(img2)

        # Concatenate encoder outputs along the channel dimension
        enc = torch.cat([enc1, enc2], dim=1)

        # Pass through decoder
        dec = self.decoder(enc)

        # Final convolution
        out = self.final_conv(dec)
        out = torch.sigmoid(out)  # Assuming binary change detection

        return out

    def get_input_size(self) -> tuple:
        """
        Specifies the input size for each image.
        
        Returns:
            tuple: Input size (channels, height, width).
        """
        return (self.in_channels, 256, 256)

    def freeze_parameters(self):
        """
        Freezes the encoder parameters to prevent them from being updated during training.
        """
        super().freeze_parameters()
        logger.info("Encoder parameters have been frozen.")

    def unfreeze_parameters(self):
        """
        Unfreezes the encoder parameters to allow them to be updated during training.
        """
        super().unfreeze_parameters()
        logger.info("Encoder parameters have been unfrozen.")
