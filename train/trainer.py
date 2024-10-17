# train/trainer.py

import logging
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.checkpoint import save_checkpoint, load_checkpoint
from train.metrics import Metric

logger = logging.getLogger(__name__)


class TrainerConfig:
    """
    Configuration parameters for the Trainer.
    """

    def __init__(
        self,
        num_epochs: int = 10,
        checkpoint_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        device: Optional[str] = None,  # 'cuda' or 'cpu'
    ):
        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path
        self.checkpoint_dir = checkpoint_dir
        self.device = device


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
        Initialize the CustomTrainer.

        Parameters
        ----------
        model : nn.Module
            The model to train.
        optimizer : torch.optim.Optimizer
            Optimizer for updating model parameters.
        loss_fn : nn.Module
            Loss function to optimize.
        metrics : List[Metric], optional
            List of metrics to evaluate.
        config : TrainerConfig, optional
            Configuration object with training parameters.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.config = config or TrainerConfig()

        # Device configuration
        self.device = torch.device(
            self.config.device if self.config.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        logger.info("Model initialized on device: %s", self.device)

        # Load checkpoint if available
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        if self.config.checkpoint_path:
            try:
                self.start_epoch, self.best_val_loss = load_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    checkpoint_path=self.config.checkpoint_path
                )
                logger.info(
                    "Checkpoint loaded from '%s' at epoch %d with best_val_loss %f",
                    self.config.checkpoint_path, self.start_epoch, self.best_val_loss
                )
            except (FileNotFoundError, ValueError) as e:
                logger.error("Failed to load checkpoint from '%s': %s", self.config.checkpoint_path, e)
                raise ValueError(f"Failed to load checkpoint from '{self.config.checkpoint_path}': {e}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """
        Train the model over the specified number of epochs.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for training data.
        val_loader : DataLoader
            DataLoader for validation data.
        """
        num_epochs = self.config.num_epochs
        logger.info("Starting training for %d epochs.", num_epochs)

        for epoch in range(self.start_epoch, num_epochs):
            logger.info("Epoch [%d/%d]", epoch + 1, num_epochs)

            train_loss = self._train_one_epoch(train_loader, epoch)
            val_loss = self._validate(val_loader, epoch)

            # Save checkpoint
            if self.config.checkpoint_dir:
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
                logger.info(
                    "Checkpoint saved for epoch %d at '%s'. Best Val Loss: %f",
                    epoch + 1, self.config.checkpoint_dir, self.best_val_loss
                )

    def _train_one_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train the model for one epoch.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader for training data.
        epoch : int
            Current epoch number.

        Returns
        -------
        float
            Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0

        # Reset metrics
        for metric in self.metrics:
            metric.reset()

        for batch_idx, batch in enumerate(train_loader):
            inputs = self._move_to_device(batch)
            targets = batch['labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self._model_forward(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Update metrics
            for metric in self.metrics:
                metric.update(outputs.detach(), targets)

            # Logging
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    "Epoch [%d/%d], Batch [%d/%d], Loss: %.4f",
                    epoch + 1, self.config.num_epochs, batch_idx + 1, len(train_loader), loss.item()
                )

        avg_loss = running_loss / len(train_loader)
        logger.info("Epoch [%d/%d] Training Loss: %.4f", epoch + 1, self.config.num_epochs, avg_loss)

        # Compute and log metrics
        for metric in self.metrics:
            metric_value = metric.compute()
            logger.info(
                "Epoch [%d/%d] Training %s: %.4f",
                epoch + 1, self.config.num_epochs, metric.__class__.__name__, metric_value
            )

        return avg_loss

    def _validate(self, val_loader: DataLoader, epoch: int) -> float:
        """
        Validate the model.

        Parameters
        ----------
        val_loader : DataLoader
            DataLoader for validation data.
        epoch : int
            Current epoch number.

        Returns
        -------
        float
            Average validation loss.
        """
        self.model.eval()
        running_loss = 0.0

        # Reset metrics
        for metric in self.metrics:
            metric.reset()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                inputs = self._move_to_device(batch)
                targets = batch['labels'].to(self.device)

                outputs = self._model_forward(inputs)
                loss = self.loss_fn(outputs, targets)
                running_loss += loss.item()

                # Update metrics
                for metric in self.metrics:
                    metric.update(outputs, targets)

        avg_loss = running_loss / len(val_loader)
        logger.info("Epoch [%d/%d] Validation Loss: %.4f", epoch + 1, self.config.num_epochs, avg_loss)

        # Compute and log metrics
        for metric in self.metrics:
            metric_value = metric.compute()
            logger.info(
                "Epoch [%d/%d] Validation %s: %.4f",
                epoch + 1, self.config.num_epochs, metric.__class__.__name__, metric_value
            )

        return avg_loss

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Move inputs to the configured device.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch data containing 'x_before' and 'x_after'.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing 'x_before' and 'x_after' tensors moved to the device.
        """
        x_before = batch['x_before'].to(self.device)
        x_after = batch['x_after'].to(self.device)
        return x_before, x_after

    def _model_forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Perform a forward pass with the model.

        Parameters
        ----------
        inputs : Tuple[torch.Tensor, torch.Tensor]
            Tuple containing 'x_before' and 'x_after' tensors.

        Returns
        -------
        torch.Tensor
            Model outputs.
        """
        x_before, x_after = inputs
        return self.model(x_before, x_after)


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
        Initialize the HuggingFaceTrainer.

        Parameters
        ----------
        model : nn.Module
            The model to train.
        optimizer : torch.optim.Optimizer
            Optimizer for updating model parameters.
        loss_fn : nn.Module
            Loss function to optimize.
        metrics : List[Metric], optional
            List of metrics to evaluate.
        config : TrainerConfig, optional
            Configuration object with training parameters.
        hf_args : Dict[str, Any], optional
            HuggingFace TrainingArguments.
        hf_train_dataset : Dataset, optional
            Training dataset for HuggingFace Trainer.
        hf_val_dataset : Dataset, optional
            Validation dataset for HuggingFace Trainer.
        """
        from transformers import Trainer, TrainingArguments

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.config = config or TrainerConfig()
        self.hf_args = hf_args or {}
        self.hf_train_dataset = hf_train_dataset
        self.hf_val_dataset = hf_val_dataset

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
        def compute_metrics(eval_pred: Tuple[Any, Any]) -> Dict[str, float]:
            """
            Compute metrics based on model predictions and targets.

            Parameters
            ----------
            eval_pred : Tuple[Any, Any]
                Tuple containing logits and labels.

            Returns
            -------
            Dict[str, float]
                Dictionary of computed metric values.
            """
            preds, labels = eval_pred
            preds = torch.tensor(preds)
            labels = torch.tensor(labels)

            # Apply threshold if necessary (assuming binary classification)
            preds = preds > 0.5

            metric_results = {}
            for metric in self.metrics:
                metric.reset()
                metric.update(preds, labels)
                metric_value = metric.compute()
                metric_name = metric.__class__.__name__.lower()
                metric_results[metric_name] = metric_value
            return metric_results

        # Initialize HuggingFace Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.hf_train_dataset,
            eval_dataset=self.hf_val_dataset,
            compute_metrics=compute_metrics,
            tokenizer=None,  # Assuming the model doesn't require tokenization
            optimizers=(self.optimizer, None),  # Let Trainer handle scheduler if needed
        )
        logger.info("HuggingFace Trainer initialized successfully.")

    def train(self) -> None:
        """
        Start the training process using HuggingFace's Trainer.
        """
        self.trainer.train()
        logger.info("HuggingFace Trainer training completed successfully.")

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model using HuggingFace's Trainer.

        Returns
        -------
        Dict[str, float]
            Dictionary of computed metric values.
        """
        eval_results = self.trainer.evaluate()
        logger.info("HuggingFace Trainer evaluation completed successfully.")
        return eval_results

    def save_best_model(self, save_path: str) -> None:
        """
        Save the best model to the specified path.

        Parameters
        ----------
        save_path : str
            Path to save the best model.
        """
        self.trainer.save_model(save_path)
        logger.info("Best model saved to '%s'.", save_path)
