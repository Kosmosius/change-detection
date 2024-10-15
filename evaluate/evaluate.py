# evaluate/evaluate.py

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass

from utils.config_parser import get_config, save_config
from utils.logger import setup_logger
from utils.checkpoint import load_checkpoint
from data.datasets import get_dataloader, get_default_transforms
from models.change_detection_transformer import ChangeDetectionTransformer
from models.siamese_unet import SiameseUNet
from train.loss_functions import get_loss_function, LossFunctionError
from train.metrics import get_metrics, MetricError
from train.trainer import TrainerError

# Initialize logger for this module
logger = logging.getLogger(__name__)


class EvaluatorError(Exception):
    """Custom exception for evaluator-related errors."""
    pass


@dataclass
class EvaluateConfig:
    """
    Data class for evaluation configuration parameters.
    """
    model_name: str
    encoder_name: str
    num_classes: int
    use_peft: bool
    peft_config: Dict[str, Any]
    in_channels: int
    out_channels: int
    feature_maps: List[int]
    evaluation_loss_function: str
    evaluation_loss_kwargs: Dict[str, Any]
    evaluation_metrics: List[str]
    metrics_threshold: float
    evaluation_batch_size: int
    evaluation_checkpoint_path: Optional[str]
    evaluation_log_interval: int
    data_eval_image_pairs: List[List[str]]
    data_eval_labels: List[str]
    data_use_s3: bool
    data_s3_bucket: Optional[str]
    data_s3_prefix: Optional[str]
    data_num_workers: int
    data_cache_transforms: bool
    data_retry_attempts: int
    data_retry_delay: float


class Evaluator:
    """
    Evaluator class to handle the evaluation of trained models.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        metrics: List['Metric'],
        config: EvaluateConfig,
    ):
        """
        Initializes the Evaluator.
        
        Args:
            model (nn.Module): The trained model to evaluate.
            loss_fn (nn.Module): The loss function used during training.
            metrics (List[Metric]): List of metrics to compute during evaluation.
            config (EvaluateConfig): Configuration object with evaluation parameters.
        """
        self.model: nn.Module = model
        self.loss_fn: nn.Module = loss_fn
        self.metrics: List['Metric'] = metrics
        self.config: EvaluateConfig = config

        # Device configuration
        self.device: torch.device = torch.device(
            self.config.evaluation_device if self.config.evaluation_device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")

        # Load checkpoint
        if self.config.evaluation_checkpoint_path and os.path.isfile(self.config.evaluation_checkpoint_path):
            try:
                _, _ = load_checkpoint(
                    model=self.model,
                    optimizer=None,  # Optimizer is not needed for evaluation
                    checkpoint_path=self.config.evaluation_checkpoint_path
                )
                logger.info(f"Checkpoint loaded from '{self.config.evaluation_checkpoint_path}'.")
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Failed to load checkpoint from '{self.config.evaluation_checkpoint_path}': {e}")
                raise EvaluatorError(f"Failed to load checkpoint from '{self.config.evaluation_checkpoint_path}': {e}") from e
        else:
            logger.error(f"Checkpoint path '{self.config.evaluation_checkpoint_path}' is invalid or does not exist.")
            raise EvaluatorError(f"Checkpoint path '{self.config.evaluation_checkpoint_path}' is invalid or does not exist.")

    def evaluate(self, eval_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Evaluates the model on the provided evaluation DataLoader.
        
        Args:
            eval_loader (DataLoader): DataLoader for evaluation data.
        
        Returns:
            Tuple[float, Dict[str, float]]: Average loss and a dictionary of metric values.
        
        Raises:
            EvaluatorError: If evaluation encounters critical issues.
        """
        self.model.eval()
        running_loss: float = 0.0

        # Reset metrics
        for metric in self.metrics:
            metric.reset()

        logger.info("Starting evaluation loop.")
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(eval_loader):
                try:
                    # Move inputs to device
                    if isinstance(inputs, (list, tuple)):
                        inputs = [input_tensor.to(self.device) for input_tensor in inputs]
                    else:
                        inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    logger.debug(f"Batch {batch_idx + 1}: Inputs and targets moved to device.")
                except Exception as e:
                    logger.error(f"Failed to move data to device during evaluation: {e}")
                    raise EvaluatorError(f"Failed to move data to device during evaluation: {e}") from e

                try:
                    # Forward pass
                    if isinstance(inputs, (list, tuple)):
                        outputs = self.model(*inputs)
                    else:
                        outputs = self.model(inputs)
                    
                    # Compute loss
                    loss = self.loss_fn(outputs, targets)
                    running_loss += loss.item()
                    logger.debug(f"Batch {batch_idx + 1}: Loss computed.")
                except (LossFunctionError, MetricError) as e:
                    logger.error(f"Error during forward pass or loss computation: {e}")
                    raise EvaluatorError(f"Error during forward pass or loss computation: {e}") from e
                except Exception as e:
                    logger.error(f"Unexpected error during evaluation: {e}")
                    raise EvaluatorError(f"Unexpected error during evaluation: {e}") from e

                try:
                    # Update metrics
                    for metric in self.metrics:
                        metric.update(outputs, targets)
                except MetricError as e:
                    logger.error(f"Error updating metrics: {e}")
                    raise

                # Logging at specified intervals
                if (batch_idx + 1) % self.config.evaluation_log_interval == 0:
                    logger.info(f"Batch [{batch_idx + 1}/{len(eval_loader)}], Loss: {loss.item():.4f}")

        # Compute average loss
        avg_loss = running_loss / len(eval_loader)
        logger.info(f"Average Evaluation Loss: {avg_loss:.4f}")

        # Compute metrics
        metric_results: Dict[str, float] = {}
        try:
            for metric in self.metrics:
                metric_value = metric.compute()
                metric_results[metric.__class__.__name__.lower()] = metric_value
                logger.info(f"Evaluation {metric.__class__.__name__}: {metric_value:.4f}")
        except MetricError as e:
            logger.error(f"Error computing metrics: {e}")
            raise EvaluatorError(f"Error computing metrics: {e}") from e

        logger.info("Evaluation process completed successfully.")
        return avg_loss, metric_results


def main() -> None:
    """
    Main function to orchestrate the evaluation process.
    """
    try:
        # Setup logger
        logger = setup_logger(__name__, log_dir="logs", log_file="evaluation.log", level=logging.INFO)
        logger.info("Logger initialized successfully.")

        # Parse configuration
        raw_config = get_config()
        config = EvaluateConfig(
            model_name=raw_config.model.name,
            encoder_name=raw_config.model.encoder_name,
            num_classes=raw_config.model.num_classes,
            use_peft=raw_config.model.use_peft,
            peft_config=raw_config.model.peft_config,
            in_channels=raw_config.model.in_channels,
            out_channels=raw_config.model.out_channels,
            feature_maps=raw_config.model.feature_maps,
            evaluation_loss_function=raw_config.evaluation.loss_function,
            evaluation_loss_kwargs=raw_config.evaluation.loss_kwargs,
            evaluation_metrics=raw_config.evaluation.metrics,
            metrics_threshold=raw_config.metrics.threshold,
            evaluation_batch_size=raw_config.evaluation.batch_size,
            evaluation_checkpoint_path=raw_config.evaluation.checkpoint_path,
            evaluation_log_interval=raw_config.evaluation.log_interval,
            data_eval_image_pairs=raw_config.data.eval_image_pairs,
            data_eval_labels=raw_config.data.eval_labels,
            data_use_s3=raw_config.data.use_s3,
            data_s3_bucket=raw_config.data.s3_bucket,
            data_s3_prefix=raw_config.data.s3_prefix,
            data_num_workers=raw_config.data.num_workers,
            data_cache_transforms=raw_config.data.cache_transforms,
            data_retry_attempts=raw_config.data.retry_attempts,
            data_retry_delay=raw_config.data.retry_delay,
        )
        logger.info("Configuration parsed successfully.")
        save_config(raw_config, "config/used_config_evaluation.yaml")  # Optionally save the used config
        logger.info("Configuration saved to 'config/used_config_evaluation.yaml'.")

        # Initialize data loader for evaluation
        eval_loader = get_dataloader(
            image_pairs=config.data_eval_image_pairs,
            labels=config.data_eval_labels,
            batch_size=config.evaluation_batch_size,
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
        logger.info("Evaluation DataLoader initialized successfully.")

        # Initialize model based on configuration
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
            logger.error(f"Unsupported model name: {config.model_name}")
            raise EvaluatorError(f"Unsupported model name: {config.model_name}")

        # Initialize loss function
        try:
            loss_fn = get_loss_function(name=config.evaluation_loss_function, **config.evaluation_loss_kwargs)
            logger.info(f"Loss function '{config.evaluation_loss_function}' initialized.")
        except LossFunctionError as e:
            logger.error(f"Failed to initialize loss function: {e}")
            raise

        # Initialize metrics
        try:
            metrics = get_metrics(names=config.evaluation_metrics, threshold=config.metrics_threshold)
            logger.info(f"Metrics initialized: {', '.join(config.evaluation_metrics)}")
        except MetricError as e:
            logger.error(f"Failed to initialize metrics: {e}")
            raise

        # Initialize Evaluator
        try:
            evaluator = Evaluator(
                model=model,
                loss_fn=loss_fn,
                metrics=metrics,
                config=config
            )
            logger.info("Evaluator initialized successfully.")
        except EvaluatorError as e:
            logger.error(f"Failed to initialize Evaluator: {e}")
            raise

        # Perform evaluation
        try:
            avg_loss, metric_results = evaluator.evaluate(eval_loader=eval_loader)
            logger.info(f"Average Evaluation Loss: {avg_loss:.4f}")
            logger.info("Evaluation Metrics:")
            for metric_name, metric_value in metric_results.items():
                logger.info(f"  {metric_name.capitalize()}: {metric_value:.4f}")
        except EvaluatorError as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    if __name__ == "__main__":
        main()
