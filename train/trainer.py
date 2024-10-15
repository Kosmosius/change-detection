# train/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass

from utils.checkpoint import save_checkpoint, load_checkpoint
from train.metrics import Metric, MetricError
from train.loss_functions import LossFunctionError

logger = logging.getLogger(__name__)


class TrainerError(Exception):
    """Custom exception for trainer-related errors."""
    pass


@dataclass
class TrainerConfig:
    """
    Configuration parameters for the Trainer.
    """
    num_epochs: int = 10
    checkpoint_path: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    device: Optional[str] = None  # 'cuda' or 'cpu'


class CustomTrainer:
    """
    Custom trainer for training and evaluating models without using HuggingFace's Trainer.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        metrics: Optional[List[Metric]] = None,
        config: Optional[TrainerConfig] = None,
    ):
        """
        Initializes the CustomTrainer.

        Args:
            model (nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            loss_fn (nn.Module): Loss function to optimize.
            metrics (List[Metric], optional): List of metrics to evaluate.
            config (TrainerConfig, optional): Configuration object with training parameters.
        """
        self.model: nn.Module = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.loss_fn: nn.Module = loss_fn
        self.metrics: List[Metric] = metrics or []
        self.config: TrainerConfig = config or TrainerConfig()

        # Device configuration
        self.device: torch.device = torch.device(
            self.config.device if self.config.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        logger.info(f"Model initialized on device: {self.device}")

        # Load checkpoint if available
        self.start_epoch: int = 0
        self.best_val_loss: float = float('inf')
        if self.config.checkpoint_path:
            try:
                self.start_epoch, self.best_val_loss = load_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    checkpoint_path=self.config.checkpoint_path
                )
                logger.info(f"Checkpoint loaded from '{self.config.checkpoint_path}' at epoch {self.start_epoch} with best_val_loss {self.best_val_loss}")
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Failed to load checkpoint from '{self.config.checkpoint_path}': {e}")
                raise TrainerError(f"Failed to load checkpoint from '{self.config.checkpoint_path}': {e}") from e

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """
        Trains the model over the specified number of epochs.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.

        Raises:
            TrainerError: If training encounters critical issues.
        """
        num_epochs = self.config.num_epochs
        logger.info(f"Starting training for {num_epochs} epochs.")

        for epoch in range(self.start_epoch, num_epochs):
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}]")
            try:
                train_loss = self._train_one_epoch(train_loader, epoch)
                val_loss = self._validate(val_loader, epoch)
            except (MetricError, LossFunctionError, TrainerError) as e:
                logger.error(f"Error during training at epoch {epoch + 1}: {e}")
                raise TrainerError(f"Error during training at epoch {epoch + 1}: {e}") from e

            # Save checkpoint
            if self.config.checkpoint_dir:
                try:
                    is_best = val_loss < self.best_val_loss
                    self.best_val_loss = min(val_loss, self.best_val_loss)
                    save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch + 1,
                        loss=val_loss,
                        checkpoint_dir=self.config.checkpoint_dir,
                        filename=f"checkpoint_epoch_{epoch + 1}.pth",
                        is_best=is_best
                    )
                    logger.info(f"Checkpoint saved for epoch {epoch + 1} at '{self.config.checkpoint_dir}'. Best Val Loss: {self.best_val_loss}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint: {e}")
                    raise TrainerError(f"Failed to save checkpoint: {e}") from e

    def _train_one_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Trains the model for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            epoch (int): Current epoch number.

        Returns:
            float: Average training loss for the epoch.

        Raises:
            MetricError: If metric updates fail.
            LossFunctionError: If loss computation fails.
        """
        self.model.train()
        running_loss: float = 0.0

        # Reset metrics
        for metric in self.metrics:
            metric.reset()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move inputs to device
            try:
                if isinstance(inputs, (list, tuple)):
                    inputs = [input_tensor.to(self.device) for input_tensor in inputs]
                else:
                    inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logger.debug(f"Batch {batch_idx + 1}: Inputs and targets moved to device.")
            except Exception as e:
                logger.error(f"Failed to move data to device: {e}")
                raise TrainerError(f"Failed to move data to device: {e}") from e

            # Forward pass
            try:
                self.optimizer.zero_grad()
                if isinstance(inputs, (list, tuple)):
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                logger.debug(f"Batch {batch_idx + 1}: Loss computed and backpropagated.")
            except (MetricError, LossFunctionError) as e:
                logger.error(f"Error during forward/backward pass: {e}")
                raise TrainerError(f"Error during forward/backward pass: {e}") from e
            except Exception as e:
                logger.error(f"Unexpected error during training: {e}")
                raise TrainerError(f"Unexpected error during training: {e}") from e

            running_loss += loss.item()

            # Update metrics
            try:
                for metric in self.metrics:
                    metric.update(outputs.detach(), targets)
            except MetricError as e:
                logger.error(f"Error updating metrics: {e}")
                raise

            # Logging
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch + 1}/{self.config.num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch [{epoch + 1}/{self.config.num_epochs}] Training Loss: {avg_loss:.4f}")

        # Compute and log metrics
        try:
            for metric in self.metrics:
                metric_value = metric.compute()
                logger.info(f"Epoch [{epoch + 1}/{self.config.num_epochs}] Training {metric.__class__.__name__}: {metric_value:.4f}")
        except MetricError as e:
            logger.error(f"Error computing metrics: {e}")
            raise

        return avg_loss

    def _validate(self, val_loader: DataLoader, epoch: int) -> float:
        """
        Validates the model.

        Args:
            val_loader (DataLoader): DataLoader for validation data.
            epoch (int): Current epoch number.

        Returns:
            float: Average validation loss.

        Raises:
            MetricError: If metric updates fail.
            LossFunctionError: If loss computation fails.
        """
        self.model.eval()
        running_loss: float = 0.0

        # Reset metrics
        for metric in self.metrics:
            metric.reset()

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                # Move inputs to device
                try:
                    if isinstance(inputs, (list, tuple)):
                        inputs = [input_tensor.to(self.device) for input_tensor in inputs]
                    else:
                        inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    logger.debug(f"Validation Batch {batch_idx + 1}: Inputs and targets moved to device.")
                except Exception as e:
                    logger.error(f"Failed to move data to device during validation: {e}")
                    raise TrainerError(f"Failed to move data to device during validation: {e}") from e

                # Forward pass
                try:
                    if isinstance(inputs, (list, tuple)):
                        outputs = self.model(*inputs)
                    else:
                        outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    running_loss += loss.item()
                    logger.debug(f"Validation Batch {batch_idx + 1}: Loss computed.")
                except (MetricError, LossFunctionError) as e:
                    logger.error(f"Error during forward pass in validation: {e}")
                    raise TrainerError(f"Error during forward pass in validation: {e}") from e
                except Exception as e:
                    logger.error(f"Unexpected error during validation: {e}")
                    raise TrainerError(f"Unexpected error during validation: {e}") from e

                # Update metrics
                try:
                    for metric in self.metrics:
                        metric.update(outputs, targets)
                except MetricError as e:
                    logger.error(f"Error updating validation metrics: {e}")
                    raise

        avg_loss = running_loss / len(val_loader)
        logger.info(f"Epoch [{epoch + 1}/{self.config.num_epochs}] Validation Loss: {avg_loss:.4f}")

        # Compute and log metrics
        try:
            for metric in self.metrics:
                metric_value = metric.compute()
                logger.info(f"Epoch [{epoch + 1}/{self.config.num_epochs}] Validation {metric.__class__.__name__}: {metric_value:.4f}")
        except MetricError as e:
            logger.error(f"Error computing validation metrics: {e}")
            raise

        return avg_loss


class HuggingFaceTrainer:
    """
    Trainer class that wraps HuggingFace's Trainer for training models.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        metrics: Optional[List[Metric]] = None,
        config: Optional[TrainerConfig] = None,
        hf_args: Optional[Dict[str, Any]] = None,
        hf_train_dataset: Optional[Any] = None,
        hf_val_dataset: Optional[Any] = None,
    ):
        """
        Initializes the HuggingFaceTrainer.

        Args:
            model (nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            loss_fn (nn.Module): Loss function to optimize.
            metrics (List[Metric], optional): List of metrics to evaluate.
            config (TrainerConfig, optional): Configuration object with training parameters.
            hf_args (Dict[str, Any], optional): HuggingFace TrainingArguments.
            hf_train_dataset (Dataset, optional): Training dataset for HuggingFace Trainer.
            hf_val_dataset (Dataset, optional): Validation dataset for HuggingFace Trainer.
        """
        from transformers import Trainer, TrainingArguments

        self.model: nn.Module = model
        self.optimizer: torch.optim.Optimizer = optimizer
        self.loss_fn: nn.Module = loss_fn
        self.metrics: List[Metric] = metrics or []
        self.config: TrainerConfig = config or TrainerConfig()
        self.hf_args: Dict[str, Any] = hf_args or {}
        self.hf_train_dataset: Optional[Any] = hf_train_dataset
        self.hf_val_dataset: Optional[Any] = hf_val_dataset

        # Prepare HuggingFace TrainingArguments
        training_args = TrainingArguments(
            output_dir=self.hf_args.get('output_dir', './results'),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.hf_args.get('per_device_train_batch_size', 8),
            per_device_eval_batch_size=self.hf_args.get('per_device_eval_batch_size', 8),
            learning_rate=self.hf_args.get('learning_rate', 5e-5),
            logging_dir=self.hf_args.get('logging_dir', './logs'),
            evaluation_strategy=self.hf_args.get('evaluation_strategy', 'epoch'),
            save_strategy=self.hf_args.get('save_strategy', 'epoch'),
            logging_steps=self.hf_args.get('logging_steps', 10),
            save_total_limit=self.hf_args.get('save_total_limit', 2),
            load_best_model_at_end=self.hf_args.get('load_best_model_at_end', True),
            metric_for_best_model=self.hf_args.get('metric_for_best_model', 'loss'),
            greater_is_better=self.hf_args.get('greater_is_better', False),
            **self.hf_args
        )

        # Define compute_metrics function
        def compute_metrics(eval_pred: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
            """
            Computes metrics based on model predictions and targets.

            Args:
                eval_pred (Tuple[torch.Tensor, torch.Tensor]): Tuple containing logits and labels.

            Returns:
                Dict[str, float]: Dictionary of computed metric values.
            """
            logits, labels = eval_pred
            logits = torch.tensor(logits)
            labels = torch.tensor(labels)

            metric_results: Dict[str, float] = {}
            for metric in self.metrics:
                try:
                    metric.reset()
                    metric.update(logits, labels)
                    metric_value = metric.compute()
                    metric_name = metric.__class__.__name__.lower()
                    metric_results[metric_name] = metric_value
                    logger.debug(f"Computed {metric_name}: {metric_value}")
                except MetricError as e:
                    logger.error(f"Error computing metric {metric.__class__.__name__}: {e}")
            return metric_results

        # Initialize HuggingFace Trainer
        try:
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.hf_train_dataset,
                eval_dataset=self.hf_val_dataset,
                compute_metrics=compute_metrics,
                tokenizer=None,  # Assuming the model doesn't require tokenization
                optimizers=(self.optimizer, None),
            )
            logger.info("HuggingFace Trainer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace Trainer: {e}")
            raise TrainerError(f"Failed to initialize HuggingFace Trainer: {e}") from e

    def train(self) -> None:
        """
        Starts the training process using HuggingFace's Trainer.

        Raises:
            TrainerError: If training encounters critical issues.
        """
        try:
            self.trainer.train()
            logger.info("HuggingFace Trainer training completed successfully.")
        except Exception as e:
            logger.error(f"Error during HuggingFace Trainer training: {e}")
            raise TrainerError(f"Error during HuggingFace Trainer training: {e}") from e

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluates the model using HuggingFace's Trainer.

        Returns:
            Dict[str, float]: Dictionary of computed metric values.

        Raises:
            TrainerError: If evaluation encounters critical issues.
        """
        try:
            eval_results = self.trainer.evaluate()
            logger.info("HuggingFace Trainer evaluation completed successfully.")
            return eval_results
        except Exception as e:
            logger.error(f"Error during HuggingFace Trainer evaluation: {e}")
            raise TrainerError(f"Error during HuggingFace Trainer evaluation: {e}") from e

    def save_best_model(self, save_path: str) -> None:
        """
        Saves the best model to the specified path.

        Args:
            save_path (str): Path to save the best model.

        Raises:
            TrainerError: If saving the model fails.
        """
        try:
            self.trainer.save_model(save_path)
            logger.info(f"Best model saved to '{save_path}'.")
        except Exception as e:
            logger.error(f"Failed to save best model to '{save_path}': {e}")
            raise TrainerError(f"Failed to save best model to '{save_path}': {e}") from e


# Example usage within a training script
if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim

    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    # Dummy dataset
    inputs = torch.randn(100, 1, 256, 256)  # Example input tensors
    targets = torch.randint(0, 2, (100, 1, 256, 256)).float()  # Example binary targets
    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Example model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x)

    model = SimpleModel()

    # Example optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    # Example metrics
    from train.metrics import Accuracy, Precision, Recall, F1Score, IoU, DiceCoefficient

    metrics = [
        Accuracy(),
        Precision(),
        Recall(),
        F1Score(),
        IoU(),
        DiceCoefficient(),
    ]

    # Trainer configuration
    config = TrainerConfig(
        num_epochs=5,
        checkpoint_dir='./checkpoints',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Initialize and start training with CustomTrainer
    try:
        trainer = CustomTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,
            config=config
        )
        trainer.train(train_loader=train_loader, val_loader=val_loader)
    except TrainerError as e:
        logger.error(f"Training failed: {e}")

    # Initialize and start training with HuggingFaceTrainer
    try:
        from transformers import TrainingArguments

        hf_args = {
            'output_dir': './hf_results',
            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 16,
            'logging_steps': 10,
            'save_total_limit': 3,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'iou',  # Example metric
        }

        # Assuming hf_train_dataset and hf_val_dataset are defined appropriately
        hf_train_dataset = TensorDataset(inputs, targets)  # Replace with actual datasets
        hf_val_dataset = TensorDataset(inputs, targets)

        hf_trainer = HuggingFaceTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,
            config=config,
            hf_args=hf_args,
            hf_train_dataset=hf_train_dataset,
            hf_val_dataset=hf_val_dataset
        )
        hf_trainer.train()
        evaluation_results = hf_trainer.evaluate()
        logger.info(f"Evaluation Results: {evaluation_results}")
        hf_trainer.save_best_model(save_path='./best_hf_model')
    except TrainerError as e:
        logger.error(f"HuggingFace Trainer failed: {e}")
