# train/train.py

import logging
from typing import Optional, List, Dict, Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from utils.config_parser import get_config, save_config
from utils.logger import setup_logger
from data.datasets import get_dataloader, get_default_transforms
from models.change_detection_transformer import ChangeDetectionTransformer
from models.siamese_unet import SiameseUNet
from train.loss_functions import get_loss_function
from train.metrics import get_metrics
from train.trainer import CustomTrainer, HuggingFaceTrainer
from utils.checkpoint import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class TrainConfig:
    """
    Data class for training configuration parameters.
    """

    def __init__(
        self,
        model_name: str,
        encoder_name: str,
        num_classes: int,
        use_peft: bool,
        peft_config: Dict[str, Any],
        in_channels: int,
        out_channels: int,
        feature_maps: Optional[List[int]],
        training_loss_function: str,
        training_loss_kwargs: Dict[str, Any],
        training_metrics: List[str],
        training_use_huggingface_trainer: bool,
        training_huggingface_args: Dict[str, Any],
        training_checkpoint_path: Optional[str],
        training_checkpoint_dir: Optional[str],
        training_num_epochs: int,
        training_learning_rate: float,
        training_batch_size: int,
        training_device: Optional[str] = None,  # 'cuda' or 'cpu'
        data_train_image_pairs: List[List[str]] = None,
        data_train_labels: List[str] = None,
        data_val_image_pairs: List[List[str]] = None,
        data_val_labels: List[str] = None,
        data_use_s3: bool = False,
        data_s3_bucket: Optional[str] = None,
        data_s3_prefix: Optional[str] = None,
        data_num_workers: int = 4,
        data_cache_transforms: bool = False,
        data_retry_attempts: int = 3,
        data_retry_delay: float = 1.0,
    ):
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.use_peft = use_peft
        self.peft_config = peft_config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_maps = feature_maps or [64, 128, 256, 512]
        self.training_loss_function = training_loss_function
        self.training_loss_kwargs = training_loss_kwargs
        self.training_metrics = training_metrics
        self.training_use_huggingface_trainer = training_use_huggingface_trainer
        self.training_huggingface_args = training_huggingface_args
        self.training_checkpoint_path = training_checkpoint_path
        self.training_checkpoint_dir = training_checkpoint_dir
        self.training_num_epochs = training_num_epochs
        self.training_learning_rate = training_learning_rate
        self.training_batch_size = training_batch_size
        self.training_device = training_device
        self.data_train_image_pairs = data_train_image_pairs
        self.data_train_labels = data_train_labels
        self.data_val_image_pairs = data_val_image_pairs
        self.data_val_labels = data_val_labels
        self.data_use_s3 = data_use_s3
        self.data_s3_bucket = data_s3_bucket
        self.data_s3_prefix = data_s3_prefix
        self.data_num_workers = data_num_workers
        self.data_cache_transforms = data_cache_transforms
        self.data_retry_attempts = data_retry_attempts
        self.data_retry_delay = data_retry_delay


def main() -> None:
    """
    Main function to orchestrate the training process.
    """
    # Setup logger
    logger = setup_logger(__name__, log_dir="logs", log_file="training.log", level=logging.INFO)
    logger.info("Logger initialized successfully.")

    # Parse configuration
    raw_config = get_config()
    config = TrainConfig(
        model_name=raw_config.model.name,
        encoder_name=raw_config.model.encoder_name,
        num_classes=raw_config.model.num_classes,
        use_peft=raw_config.model.use_peft,
        peft_config=raw_config.model.peft_config,
        in_channels=raw_config.model.in_channels,
        out_channels=raw_config.model.out_channels,
        feature_maps=raw_config.model.feature_maps,
        training_loss_function=raw_config.training.loss_function,
        training_loss_kwargs=raw_config.training.loss_kwargs,
        training_metrics=raw_config.training.metrics,
        training_use_huggingface_trainer=raw_config.training.use_huggingface_trainer,
        training_huggingface_args=raw_config.training.huggingface_args,
        training_checkpoint_path=raw_config.training.checkpoint_path,
        training_checkpoint_dir=raw_config.training.checkpoint_dir,
        training_num_epochs=raw_config.training.num_epochs,
        training_learning_rate=raw_config.training.learning_rate,
        training_batch_size=raw_config.training.batch_size,
        training_device=raw_config.training.device,
        data_train_image_pairs=raw_config.data.train_image_pairs,
        data_train_labels=raw_config.data.train_labels,
        data_val_image_pairs=raw_config.data.val_image_pairs,
        data_val_labels=raw_config.data.val_labels,
        data_use_s3=raw_config.data.use_s3,
        data_s3_bucket=raw_config.data.s3_bucket,
        data_s3_prefix=raw_config.data.s3_prefix,
        data_num_workers=raw_config.data.num_workers,
        data_cache_transforms=raw_config.data.cache_transforms,
        data_retry_attempts=raw_config.data.retry_attempts,
        data_retry_delay=raw_config.data.retry_delay,
    )
    logger.info("Configuration parsed successfully.")
    save_config(raw_config, "config/used_config.yaml")  # Optionally save the used config
    logger.info("Configuration saved to 'config/used_config.yaml'.")

    # Initialize data loaders
    train_loader = get_dataloader(
        image_pairs=config.data_train_image_pairs,
        labels=config.data_train_labels,
        batch_size=config.training_batch_size,
        shuffle=True,
        transform=get_default_transforms(),
        use_s3=config.data_use_s3,
        s3_bucket=config.data_s3_bucket,
        s3_prefix=config.data_s3_prefix,
        num_workers=config.data_num_workers,
        cache_transforms=config.data_cache_transforms,
        retry_attempts=config.data_retry_attempts,
        retry_delay=config.data_retry_delay
    )
    logger.info("Training DataLoader initialized successfully.")

    val_loader = get_dataloader(
        image_pairs=config.data_val_image_pairs,
        labels=config.data_val_labels,
        batch_size=config.training_batch_size,
        shuffle=False,
        transform=get_default_transforms(),
        use_s3=config.data_use_s3,
        s3_bucket=config.data_s3_bucket,
        s3_prefix=config.data_s3_prefix,
        num_workers=config.data_num_workers,
        cache_transforms=config.data_cache_transforms,
        retry_attempts=config.data_retry_attempts,
        retry_delay=config.data_retry_delay
    )
    logger.info("Validation DataLoader initialized successfully.")

    # Initialize model
    if config.model_name.lower() == "change_detection_transformer":
        model = ChangeDetectionTransformer(
            encoder_name=config.encoder_name,
            num_classes=config.num_classes,
            use_peft=config.use_peft,
            peft_config=config.peft_config
        )
        logger.info("ChangeDetectionTransformer model initialized.")
    elif config.model_name.lower() == "siamese_unet":
        model = SiameseUNet(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            feature_maps=config.feature_maps
        )
        logger.info("SiameseUNet model initialized.")
    else:
        logger.error("Unsupported model name: %s", config.model_name)
        raise ValueError(f"Unsupported model name: {config.model_name}")

    # Count trainable parameters
    if hasattr(model, 'count_parameters'):
        model.count_parameters()
    else:
        logger.warning("Model does not have a 'count_parameters' method.")

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.training_learning_rate)
    logger.info("Optimizer initialized with learning rate: %f", config.training_learning_rate)

    # Initialize loss function
    loss_fn = get_loss_function(name=config.training_loss_function, **config.training_loss_kwargs)
    logger.info("Loss function '%s' initialized.", config.training_loss_function)

    # Initialize metrics
    metrics = get_metrics(names=config.training_metrics, threshold=raw_config.metrics.threshold)
    logger.info("Metrics initialized: %s", ', '.join(config.training_metrics))

    # Initialize trainer
    trainer_config = {
        'num_epochs': config.training_num_epochs,
        'checkpoint_path': config.training_checkpoint_path,
        'checkpoint_dir': config.training_checkpoint_dir,
        'device': config.training_device
    }

    if config.training_use_huggingface_trainer:
        trainer = HuggingFaceTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,
            config=trainer_config,
            hf_args=config.training_huggingface_args,
            hf_train_dataset=train_loader.dataset,
            hf_val_dataset=val_loader.dataset
        )
        logger.info("HuggingFaceTrainer initialized successfully.")
    else:
        trainer = CustomTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,
            config=trainer_config
        )
        logger.info("CustomTrainer initialized successfully.")

    # Start training
    trainer.train(train_loader, val_loader)
    logger.info("Training process completed successfully.")


if __name__ == "__main__":
    main()
