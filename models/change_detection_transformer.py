# models/change_detection_transformer.py

import logging
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

from .base_model import BaseModel
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
            peft_config (Optional[Dict[str, Any]]): Configuration for PEFT.
            decoder_config (Optional[Dict[str, Any]]): Configuration for the decoder layers.
            device (Optional[torch.device]): Device to run the model on.
                If None, uses the default device from BaseModel.

        Raises:
            ValueError: If invalid parameters are provided.
        """
        super().__init__(device=device)
        self.encoder_name: str = encoder_name
        self.num_classes: int = num_classes
        self.use_peft: bool = use_peft
        self.peft_config: Optional[Dict[str, Any]] = peft_config
        self.decoder_config: Dict[str, Any] = decoder_config or {}

        if self.num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")

        # Load pre-trained HuggingFace encoder
        self.encoder = load_huggingface_model(
            model_name=self.encoder_name,
            pretrained=True,
            config_kwargs={'output_hidden_states': False},
        )
        logger.info("Loaded HuggingFace encoder '%s'.", self.encoder_name)

        # Modify the encoder to accept 6-channel input
        self._modify_encoder_input_channels()

        # Apply PEFT if enabled
        if self.use_peft:
            if not self.peft_config:
                raise ValueError("PEFT configuration must be provided when use_peft is True.")
            self.encoder = apply_peft(model=self.encoder, peft_config=self.peft_config)
            logger.info("Applied PEFT to the encoder.")

        # Freeze encoder parameters if not using PEFT
        if not self.use_peft:
            self.freeze_encoder()
            logger.debug("Encoder parameters have been frozen.")

        # Define decoder layers based on decoder_config
        self.decoder: nn.Module = self._build_decoder()
        logger.info("Initialized decoder layers.")

    def _modify_encoder_input_channels(self) -> None:
        """
        Modifies the encoder's first convolutional layer to accept 6 input channels instead of 3.
        Initializes the new channels by copying the weights from the original 3 channels.
        """
        # Access the encoder's patch embedding layer
        patch_embeddings = self.encoder.embeddings.patch_embeddings

        # Helper function to find the first Conv2d layer recursively
        def find_conv_layer(module: nn.Module) -> Optional[Tuple[str, nn.Conv2d]]:
            for name, child in module.named_children():
                if isinstance(child, nn.Conv2d):
                    return name, child
                else:
                    result = find_conv_layer(child)
                    if result is not None:
                        return result
            return None

        conv_info = find_conv_layer(patch_embeddings)

        if conv_info is None:
            logger.error(
                f"Failed to find a convolutional layer in patch_embeddings. "
                f"Available attributes: {list(patch_embeddings.__dict__.keys())}"
            )
            raise ValueError(
                "Encoder does not have a Conv2d layer in patch_embeddings. "
                "Please verify the encoder architecture."
            )

        name, conv_layer = conv_info
        logger.debug(f"Found convolutional layer '{name}' in patch_embeddings.")

        # Create a new Conv2d layer with 6 input channels
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=conv_layer.out_channels,
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            bias=conv_layer.bias is not None
        )

        # Initialize the new_conv's weights
        with torch.no_grad():
            # Copy weights for the original 3 channels
            new_conv.weight[:, :3, :, :] = conv_layer.weight
            # Initialize the new 3 channels by copying the original weights
            new_conv.weight[:, 3:, :, :] = conv_layer.weight
            # If bias exists, copy it
            if conv_layer.bias is not None:
                new_conv.bias = conv_layer.bias

        # Replace the original conv layer with the new conv layer
        setattr(patch_embeddings, name, new_conv)
        logger.info(f"Modified encoder's patch_embeddings.{name} to accept 6-channel input.")

    def _build_decoder(self) -> nn.Module:
        """
        Builds the decoder part of the model based on the provided configuration.

        Returns:
            nn.Module: The decoder module.

        Raises:
            ValueError: If the decoder configuration is invalid.
        """
        hidden_size = self.encoder.config.hidden_size
        layers = []

        # Decoder configuration handling
        num_layers = self.decoder_config.get('num_layers', 2)
        layer_size = self.decoder_config.get('layer_size', 256)
        dropout_rate = self.decoder_config.get('dropout_rate', 0.1)
        activation = self.decoder_config.get('activation', 'relu').lower()

        activations = {
            'relu': nn.ReLU(inplace=True),
            'leakyrelu': nn.LeakyReLU(inplace=True),
            'gelu': nn.GELU(),
        }

        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, layer_size))
            act = activations.get(activation, nn.ReLU(inplace=True))
            layers.append(act)
            layers.append(nn.Dropout(dropout_rate))
            hidden_size = layer_size  # Update for next layer

        layers.append(nn.Linear(hidden_size, self.num_classes))
        return nn.Sequential(*layers)

    def forward(self, x_before: torch.Tensor, x_after: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x_before (torch.Tensor): The 'before' image tensor of shape (batch_size, channels, height, width).
            x_after (torch.Tensor): The 'after' image tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width).

        Raises:
            ValueError: If input tensors have incompatible shapes.
        """
        # Validate input shapes
        if x_before.shape != x_after.shape:
            raise ValueError("Input tensors x_before and x_after must have the same shape.")
        if x_before.dim() != 4:
            raise ValueError("Input tensors must be 4-dimensional (batch_size, channels, height, width).")

        # Concatenate before and after images along the channel dimension
        x = torch.cat((x_before, x_after), dim=1)  # Shape: (batch_size, 6, H, W)

        # Prepare inputs for the encoder
        # Assuming the encoder expects inputs under the key 'pixel_values'
        encoder_outputs = self.encoder(pixel_values=x)
        pooled_output = encoder_outputs.last_hidden_state[:, 0]  # CLS token, Shape: (batch_size, hidden_size)

        # Pass through decoder
        logits = self.decoder(pooled_output)  # Shape: (batch_size, num_classes)
        logits = logits.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, num_classes, 1, 1)

        # Upsample logits to match input dimensions
        logits = nn.functional.interpolate(
            logits, size=x_before.shape[2:], mode='bilinear', align_corners=False
        )  # Shape: (batch_size, num_classes, H, W)

        return logits

    def get_input_size(self) -> Tuple[int, int, int]:
        """
        Returns the expected input size for the model.

        Returns:
            Tuple[int, int, int]: Input size in the format (channels, height, width).
        """
        # The input size depends on the encoder's expected input size
        # For 'google/vit-base-patch16-224', the input size is (6, 224, 224)
        # Since we concatenate before and after images, channels are doubled
        return (6, 224, 224)

    def freeze_encoder(self) -> None:
        """
        Freezes the encoder parameters to prevent them from being updated during training.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.debug("Encoder parameters have been frozen.")

    def unfreeze_encoder(self) -> None:
        """
        Unfreezes the encoder parameters to allow them to be updated during training.
        """
        for param in self.encoder.parameters():
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
            summary(self, input_size=(1, *input_size))
        except Exception as e:
            logger.error("Failed to generate model summary: %s", e)
            raise
