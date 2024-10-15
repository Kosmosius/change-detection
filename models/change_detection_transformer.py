# models/change_detection_transformer.py

import torch
import torch.nn as nn
import logging
from models.base_model import BaseModel
from transformers import AutoModel, AutoConfig
from utils.huggingface_utils import apply_peft

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
        peft_config: dict = None
    ):
        """
        Initializes the ChangeDetectionTransformer model.
        
        Args:
            encoder_name (str): HuggingFace model name for the encoder.
            num_classes (int): Number of output classes.
            use_peft (bool): Whether to apply Parameter-Efficient Fine-Tuning.
            peft_config (dict, optional): Configuration for PEFT.
        """
        super(ChangeDetectionTransformer, self).__init__()
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.use_peft = use_peft
        self.peft_config = peft_config

        # Load pre-trained HuggingFace encoder
        self.encoder = AutoModel.from_pretrained(self.encoder_name)
        logger.info(f"Loaded HuggingFace encoder '{self.encoder_name}'.")

        # Freeze encoder parameters if not using PEFT
        if not self.use_peft:
            self.freeze_parameters()

        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_classes)
        )
        logger.info("Initialized decoder layers.")

        # Apply PEFT if enabled
        if self.use_peft and self.peft_config:
            self.encoder = apply_peft(self.encoder, self.peft_config)
            logger.info("Applied PEFT to the encoder.")

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes, height, width).
        """
        # Extract features from the encoder
        outputs = self.encoder(pixel_values=x)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token

        # Pass through decoder
        logits = self.decoder(pooled_output)
        logits = logits.unsqueeze(-1).unsqueeze(-1)  # Reshape to (batch_size, num_classes, 1, 1)

        # For change detection, we might need to upscale or process logits further
        # This is a placeholder for actual implementation
        return logits

    def get_input_size(self) -> tuple:
        """
        Specifies the input size for the model.
        
        Returns:
            tuple: Input size (channels, height, width).
        """
        return (3, 224, 224)  # Typical input size for ViT models

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
