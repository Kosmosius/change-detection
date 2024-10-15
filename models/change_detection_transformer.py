# models/change_detection_transformer.py

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any, Tuple, Union

from models.base_model import BaseModel
from transformers import AutoModel, AutoConfig
from utils.huggingface_utils import apply_peft, load_huggingface_model

logger = logging.getLogger(__name__)


class ChangeDetectionTransformer(BaseModel):
    """
    Transformer-based model for Change Detection tasks.
    Utilizes a pre-trained HuggingFace Vision Transformer as the encoder.
    """

    def __init__(
        self,
        encoder_name: str = "google/vit-base-patch16-224",
        num_classes: int = 1,
        use_peft: bool = False,
        peft_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initializes the ChangeDetectionTransformer model.

        Args:
            encoder_name (str): HuggingFace model name for the encoder.
            num_classes (int): Number of output classes.
            use_peft (bool): Whether to apply Parameter-Efficient Fine-Tuning.
            peft_config (Dict[str, Any], optional): Configuration for PEFT.
            decoder_config (Dict[str, Any], optional): Configuration for the decoder layers.
            device (torch.device, optional): Device to run the model on.
                If None, uses the default device from BaseModel.
        """
        super(ChangeDetectionTransformer, self).__init__(device=device)
        self.encoder_name: str = encoder_name
        self.num_classes: int = num_classes
        self.use_peft: bool = use_peft
        self.peft_config: Optional[Dict[str, Any]] = peft_config
        self.decoder_config: Dict[str, Any] = decoder_config or {}

        # Load pre-trained HuggingFace encoder
        try:
            self.encoder: AutoModel = load_huggingface_model(
                model_name=self.encoder_name,
                pretrained=True,
                config_kwargs={'output_hidden_states': False},
            )
            logger.info(f"Loaded HuggingFace encoder '{self.encoder_name}'.")
        except ValueError as e:
            logger.error(f"Failed to load encoder '{self.encoder_name}': {e}")
            raise

        # Apply PEFT if enabled
        if self.use_peft:
            if not self.peft_config:
                logger.error("PEFT is enabled but no PEFT configuration provided.")
                raise ValueError("PEFT configuration must be provided when use_peft is True.")
            try:
                self.encoder = apply_peft(model=self.encoder, peft_config=self.peft_config)
                logger.info("Applied PEFT to the encoder.")
            except ValueError as e:
                logger.error(f"Failed to apply PEFT: {e}")
                raise

        # Freeze encoder parameters if not using PEFT
        if not self.use_peft:
            self.freeze_parameters()
            logger.info("Encoder parameters have been frozen.")

        # Define decoder layers based on decoder_config
        self.decoder: nn.Module = self._build_decoder()
        logger.info("Initialized decoder layers.")

    def _build_decoder(self) -> nn.Module:
        """
        Builds the decoder part of the model based on the provided configuration.

        Returns:
            nn.Module: The decoder module.

        Raises:
            ValueError: If the decoder configuration is invalid.
        """
        try:
            hidden_size = self.encoder.config.hidden_size
            layers = []

            # Example decoder configuration handling
            num_layers = self.decoder_config.get('num_layers', 2)
            layer_size = self.decoder_config.get('layer_size', 256)
            dropout_rate = self.decoder_config.get('dropout_rate', 0.1)
            activation = self.decoder_config.get('activation', 'relu')

            for _ in range(num_layers):
                layers.append(nn.Linear(hidden_size, layer_size))
                if activation.lower() == 'relu':
                    layers.append(nn.ReLU(inplace=True))
                elif activation.lower() == 'leakyrelu':
                    layers.append(nn.LeakyReLU(inplace=True))
                else:
                    logger.warning(f"Unsupported activation '{activation}'. Defaulting to ReLU.")
                    layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout_rate))
                hidden_size = layer_size  # Update for next layer

            layers.append(nn.Linear(hidden_size, self.num_classes))
            return nn.Sequential(*layers)
        except Exception as e:
            logger.error(f"Failed to build decoder: {e}")
            raise ValueError("Invalid decoder configuration.") from e

    def forward(self, x_before: torch.Tensor, x_after: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x_before (torch.Tensor): The 'before' image tensor of shape (batch_size, channels, height, width).
            x_after (torch.Tensor): The 'after' image tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width).
        """
        try:
            # Concatenate before and after images along the channel dimension
            x = torch.cat((x_before, x_after), dim=1)  # Shape: (batch_size, 6, H, W)

            # Extract features from the encoder
            encoder_outputs = self.encoder(pixel_values=x)
            pooled_output = encoder_outputs.last_hidden_state[:, 0]  # CLS token, Shape: (batch_size, hidden_size)

            # Pass through decoder
            logits = self.decoder(pooled_output)  # Shape: (batch_size, num_classes)
            logits = logits.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, num_classes, 1, 1)

            # Upsample logits to match input dimensions (assuming input height and width are multiples of 1)
            logits = nn.functional.interpolate(logits, size=x_before.shape[2:], mode='bilinear', align_corners=False)

            return logits
        except Exception as e:
            logger.error(f"Error during forward pass: {e}")
            raise

    def get_input_size(self) -> Tuple[int, int, int]:
        """
        Specifies the input size for the model.

        Returns:
            Tuple[int, int, int]: Input size in the format (channels, height, width).

        Raises:
            NotImplementedError: If input size is not defined.
        """
        return (6, 224, 224)  # 6 channels for concatenated before and after images

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

    def summary(self) -> None:
        """
        Prints a summary of the model architecture and parameter count.
        """
        try:
            super().summary()
            logger.info("Model summary generated successfully.")
        except Exception as e:
            logger.error(f"Failed to generate model summary: {e}")

