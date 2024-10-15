# evaluate/evaluate.py

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.config_parser import get_config, save_config
from utils.logger import setup_logger
from utils.checkpoint import load_checkpoint
from data.datasets import get_dataloader, get_default_transforms
from models.change_detection_transformer import ChangeDetectionTransformer
from models.siamese_unet import SiameseUNet
from train.loss_functions import get_loss_function
from train.metrics import get_metrics
import logging

def evaluate():
    """
    Evaluates the trained model on the validation/test dataset.
    """
    # Setup logger
    logger = setup_logger(__name__, log_dir="logs", log_file="evaluation.log", level=logging.INFO)
    logger.info("Starting evaluation process...")
    
    try:
        # Parse configuration
        config = get_config()
        save_config(config, "config/used_config_evaluation.yaml")  # Optionally save the used config
        
        # Initialize data loader for evaluation
        eval_loader = get_dataloader(
            image_pairs=config.data.eval_image_pairs,  # Ensure this key exists in config
            labels=config.data.eval_labels,
            batch_size=config.evaluation.batch_size,
            shuffle=False,
            transform=get_default_transforms(),
            use_s3=config.data.use_s3,
            s3_bucket=config.data.s3_bucket,
            s3_prefix=config.data.s3_prefix,
            num_workers=config.data.num_workers
        )
        logger.info("Initialized evaluation data loader.")
        
        # Initialize model based on configuration
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"Model moved to device: {device}")
        
        # Load checkpoint
        if config.evaluation.checkpoint_path and os.path.isfile(config.evaluation.checkpoint_path):
            start_epoch, previous_loss = load_checkpoint(model, None, config.evaluation.checkpoint_path)
            logger.info(f"Loaded checkpoint from '{config.evaluation.checkpoint_path}' at epoch {start_epoch} with loss {previous_loss:.4f}.")
        else:
            logger.error(f"Checkpoint path '{config.evaluation.checkpoint_path}' is invalid or does not exist.")
            return
        
        # Initialize loss function
        loss_fn = get_loss_function(name=config.evaluation.loss_function, **config.evaluation.loss_kwargs)
        logger.info(f"Initialized loss function: {config.evaluation.loss_function}")
        
        # Initialize metrics
        metrics = get_metrics(names=config.evaluation.metrics, threshold=config.metrics.threshold)
        logger.info(f"Initialized metrics: {[metric.__class__.__name__ for metric in metrics]}")
        
        # Set model to evaluation mode
        model.eval()
        
        # Initialize variables to track evaluation
        total_loss = 0.0
        for metric in metrics:
            metric.reset()
        
        # Evaluation loop
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(eval_loader):
                if isinstance(inputs, list) or isinstance(inputs, tuple):
                    inputs = [input_tensor.to(device) for input_tensor in inputs]
                else:
                    inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                if isinstance(inputs, list) or isinstance(inputs, tuple):
                    outputs = model(*inputs)
                else:
                    outputs = model(inputs)
                
                # Compute loss
                loss = loss_fn(outputs, targets)
                total_loss += loss.item()
                
                # Update metrics
                for metric in metrics:
                    metric.update(outputs, targets)
                
                if batch_idx % config.evaluation.log_interval == 0:
                    logger.info(f"Batch [{batch_idx}/{len(eval_loader)}], Loss: {loss.item():.4f}")
        
        # Compute average loss
        avg_loss = total_loss / len(eval_loader)
        logger.info(f"Average Evaluation Loss: {avg_loss:.4f}")
        
        # Compute and log metrics
        for metric in metrics:
            metric_value = metric.compute()
            logger.info(f"Evaluation {metric.__class__.__name__}: {metric_value:.4f}")
        
        logger.info("Evaluation process completed successfully.")
    
    if __name__ == "__main__":
        evaluate()
