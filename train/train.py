# train/train.py

import os
import torch
from torch import nn, optim
from utils.config_parser import get_config, save_config
from utils.logger import setup_logger
from data.datasets import get_dataloader, get_default_transforms
from models.change_detection_transformer import ChangeDetectionTransformer
from models.siamese_unet import SiameseUNet
from train.loss_functions import get_loss_function
from train.metrics import get_metrics
from train.trainer import CustomTrainer, HuggingFaceTrainer
from utils.checkpoint import save_checkpoint, load_checkpoint
import logging

def main():
    # Setup logger
    logger = setup_logger(__name__, log_dir="logs", log_file="training.log", level=logging.INFO)

    # Parse configuration
    config = get_config()
    save_config(config, "config/used_config.yaml")  # Optionally save the used config

    # Initialize data loaders
    train_loader = get_dataloader(
        image_pairs=config.data.train_image_pairs,
        labels=config.data.train_labels,
        batch_size=config.training.batch_size,
        shuffle=True,
        transform=get_default_transforms(),
        use_s3=config.data.use_s3,
        s3_bucket=config.data.s3_bucket,
        s3_prefix=config.data.s3_prefix,
        num_workers=config.data.num_workers
    )

    val_loader = get_dataloader(
        image_pairs=config.data.val_image_pairs,
        labels=config.data.val_labels,
        batch_size=config.training.batch_size,
        shuffle=False,
        transform=get_default_transforms(),
        use_s3=config.data.use_s3,
        s3_bucket=config.data.s3_bucket,
        s3_prefix=config.data.s3_prefix,
        num_workers=config.data.num_workers
    )

    # Initialize model
    if config.model.name == "change_detection_transformer":
        model = ChangeDetectionTransformer(
            encoder_name=config.model.encoder_name,
            num_classes=config.model.num_classes,
            use_peft=config.model.use_peft,
            peft_config=config.model.peft_config
        )
    elif config.model.name == "siamese_unet":
        model = SiameseUNet(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            feature_maps=config.model.feature_maps
        )
    else:
        logger.error(f"Unsupported model name: {config.model.name}")
        return

    # Move model to device
    model.to_device()

    # Count trainable parameters
    model.count_parameters()

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Initialize loss function
    loss_fn = get_loss_function(name=config.training.loss_function, **config.training.loss_kwargs)

    # Initialize metrics
    metrics = get_metrics(names=config.training.metrics, threshold=config.metrics.threshold)

    # Initialize trainer
    if config.training.use_huggingface_trainer:
        trainer = HuggingFaceTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,
            config=config,
            hf_args=config.training.huggingface_args,
            hf_train_dataset=train_loader.dataset,
            hf_val_dataset=val_loader.dataset
        )
    else:
        trainer = CustomTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,
            config=config
        )

    # Attempt to load from a checkpoint
    if config.training.checkpoint_path:
        start_epoch, previous_loss = load_checkpoint(model, optimizer, config.training.checkpoint_path)
    else:
        start_epoch = 0

    # Start training
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
