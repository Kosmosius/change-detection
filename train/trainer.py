# train/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from typing import Optional, List, Dict, Any
from utils.checkpoint import save_checkpoint, load_checkpoint
from transformers import Trainer as HfTrainer, TrainingArguments
from train.metrics import Metric

logger = logging.getLogger(__name__)

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
        config: Any = None,
    ):
        """
        Initializes the CustomTrainer.
        
        Args:
            model (nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            loss_fn (nn.Module): Loss function to optimize.
            metrics (List[Metric], optional): List of metrics to evaluate.
            config (Any, optional): Configuration object with training parameters.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Load checkpoint if available
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        if config and getattr(config.training, 'checkpoint_path', None):
            self.start_epoch, self.best_val_loss = load_checkpoint(
                self.model, self.optimizer, config.training.checkpoint_path
            )

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Trains the model.
        
        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
        """
        num_epochs = self.config.training.num_epochs if self.config else 10

        for epoch in range(self.start_epoch, num_epochs):
            logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
            train_loss = self._train_one_epoch(train_loader, epoch)
            val_loss = self._validate(val_loader, epoch)

            # Save checkpoint
            if self.config and getattr(self.config.training, 'checkpoint_dir', None):
                is_best = val_loss < self.best_val_loss
                self.best_val_loss = min(val_loss, self.best_val_loss)
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    self.config.training.checkpoint_dir,
                    is_best=is_best
                )

    def _train_one_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Trains the model for one epoch.
        
        Args:
            train_loader (DataLoader): DataLoader for training data.
            epoch (int): Current epoch number.
        
        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0

        # Reset metrics
        for metric in self.metrics:
            metric.reset()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                inputs = [input_tensor.to(self.device) for input_tensor in inputs]
            else:
                inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                outputs = self.model(*inputs)
            else:
                outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Update metrics
            for metric in self.metrics:
                metric.update(outputs.detach(), targets)

            if batch_idx % 10 == 0:
                logger.info(f"Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        logger.info(f"Training Loss: {avg_loss:.4f}")

        # Compute metrics
        for metric in self.metrics:
            metric_value = metric.compute()
            logger.info(f"Training {metric.__class__.__name__}: {metric_value:.4f}")

        return avg_loss

    def _validate(self, val_loader: DataLoader, epoch: int) -> float:
        """
        Validates the model.
        
        Args:
            val_loader (DataLoader): DataLoader for validation data.
            epoch (int): Current epoch number.
        
        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        running_loss = 0.0

        # Reset metrics
        for metric in self.metrics:
            metric.reset()

        with torch.no_grad():
            for inputs, targets in val_loader:
                if isinstance(inputs, list) or isinstance(inputs, tuple):
                    inputs = [input_tensor.to(self.device) for input_tensor in inputs]
                else:
                    inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                if isinstance(inputs, list) or isinstance(inputs, tuple):
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                running_loss += loss.item()

                # Update metrics
                for metric in self.metrics:
                    metric.update(outputs, targets)

        avg_loss = running_loss / len(val_loader)
        logger.info(f"Validation Loss: {avg_loss:.4f}")

        # Compute metrics
        for metric in self.metrics:
            metric_value = metric.compute()
            logger.info(f"Validation {metric.__class__.__name__}: {metric_value:.4f}")

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
        config: Any = None,
        hf_args: Optional[Dict] = None,
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
            config (Any, optional): Configuration object with training parameters.
            hf_args (Dict, optional): HuggingFace TrainingArguments.
            hf_train_dataset (Dataset, optional): Training dataset for HuggingFace Trainer.
            hf_val_dataset (Dataset, optional): Validation dataset for HuggingFace Trainer.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics or []
        self.config = config
        self.hf_args = hf_args or {}
        self.hf_train_dataset = hf_train_dataset
        self.hf_val_dataset = hf_val_dataset

        # Prepare HuggingFace TrainingArguments
        training_args = TrainingArguments(
            output_dir=self.hf_args.get('output_dir', './results'),
            num_train_epochs=self.config.training.num_epochs if self.config else 10,
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
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            logits = torch.tensor(logits)
            labels = torch.tensor(labels)
            metric_results = {}
            for metric in self.metrics:
                metric.reset()
                metric.update(logits, labels)
                metric_value = metric.compute()
                metric_name = metric.__class__.__name__.lower()
                metric_results[metric_name] = metric_value
            return metric_results

        # Initialize HuggingFace Trainer
        self.trainer = HfTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.hf_train_dataset,
            eval_dataset=self.hf_val_dataset,
            compute_metrics=compute_metrics,
            tokenizer=None,  # Assuming the model doesn't require tokenization
            optimizers=(self.optimizer, None),
        )

    def train(self):
        """
        Starts the training process using HuggingFace's Trainer.
        """
        self.trainer.train()
