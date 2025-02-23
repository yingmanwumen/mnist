"""
This module implements the training pipeline for the MNIST digit classification model.
It includes:
- Hyperparameter configuration
- Training loop with early stopping
- Model evaluation on validation and test sets
- Multiple evaluation metrics (accuracy, precision, recall, F1)
- Best model checkpoint saving
"""

from data import DataSet
from model import Model
import torch
from dataclasses import dataclass
from torch import nn
from torch import optim
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
from tabulate import tabulate
from typing import Dict, Any


@dataclass
class Hyperparameters:
    """
    Configuration class for training hyperparameters.

    Attributes:
        batch_size (int): Number of samples per training batch
        num_epochs (int): Maximum number of training epochs
    """

    batch_size: int
    num_epochs: int


class Train(object):
    """
    Main training class that handles the complete training pipeline.

    Features:
    - Automatic device selection (GPU/CPU)
    - Cross-entropy loss and NAdam optimizer
    - Multiple evaluation metrics
    - Early stopping based on validation accuracy
    - Best model checkpoint saving

    Args:
        hyperparameters (Hyperparameters): Training configuration
        model (Model): Neural network model to train
    """

    def __init__(self, hyperparameters: Hyperparameters, model: Model):
        """
        Initialize the training environment.

        Sets up:
        - Dataset with specified batch size
        - Device (CPU/GPU) configuration
        - Loss function and optimizer
        - Evaluation metrics for train/validation/test
        """
        self.hyperparameters = hyperparameters
        self.dataset = DataSet(batch_size=hyperparameters.batch_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.NAdam(self.model.parameters())

        # Initialize evaluation metrics for multi-class classification
        metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="multiclass", num_classes=10),
                "precision": Precision(
                    task="multiclass", num_classes=10, average="macro"
                ),
                "recall": Recall(task="multiclass", num_classes=10, average="macro"),
                "f1": F1Score(task="multiclass", num_classes=10, average="macro"),
            }
        ).to(self.device)

        self.train_metrics = metrics.clone()
        self.validate_metrics = metrics.clone()
        self.test_metrics = metrics.clone()

    def train(self):
        """
        Execute the complete training pipeline.

        Features:
        - Epoch-wise training with loss tracking
        - Regular validation checks
        - Early stopping if validation accuracy doesn't improve
        - Best model checkpoint saving
        - Final evaluation on test set
        """
        # Initialize early stopping parameters
        best_val_accuracy = 0.0  # Track best validation accuracy
        epochs_no_improve = 0  # Counter for epochs without improvement
        patience = 5  # Number of epochs to wait before early stopping

        for epoch in range(self.hyperparameters.num_epochs):
            loss, train_metrics = self._epoch()
            print(f"\nEpoch: {epoch + 1}/{self.hyperparameters.num_epochs}")
            print(f"Loss: {loss:.4f}")
            print("\nTrain metrics:")
            print(self._format_metrics(train_metrics))

            validate_metrics = self._validate()
            print("\nValidate metrics:")
            print(self._format_metrics(validate_metrics))
            print()

            current_val_accuracy = validate_metrics.get("accuracy", 0.0)
            if current_val_accuracy > best_val_accuracy:
                best_val_accuracy = current_val_accuracy
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), "best_model.pth")
                print("Model improved. Saved current best model.")
            else:
                epochs_no_improve += 1
                print(
                    f"Validation accuracy did not improve for {epochs_no_improve} epoch(s)"
                )

            if epochs_no_improve >= patience:
                print(
                    f"Early stopping triggered. No improvement in validation accuracy for {patience} consecutive epochs."
                )
                break

        self.model.load_state_dict(torch.load("best_model.pth"))
        test_metrics = self.test()
        print("\nTest metrics:")
        print(self._format_metrics(test_metrics))
        print()

    def _format_metrics(self, metrics_dict: Dict[str, Any]) -> str:
        """
        Format metrics dictionary into a human-readable table.

        Args:
            metrics_dict (Dict[str, Any]): Dictionary containing metric names and values

        Returns:
            str: Formatted table string of metrics
        """
        formatted_metrics = {k: f"{float(v):.4f}" for k, v in metrics_dict.items()}
        headers = ["Metric", "Value"]
        data = [[k, v] for k, v in formatted_metrics.items()]

        return tabulate(data, headers=headers, tablefmt="pretty")

    def _epoch(self):
        """
        Execute one training epoch.

        Performs:
        - Forward pass
        - Loss calculation
        - Backward pass
        - Optimizer step
        - Metrics update

        Returns:
            tuple: (average_loss, computed_metrics)
        """
        # Set model to training mode (enables dropout, batch norm updates, etc.)
        self.model.train()
        self.train_metrics.reset()

        total_loss = 0
        for data, labels in self.dataset.train_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            self.train_metrics.update(outputs, labels)
        return total_loss / len(self.dataset.train_loader), self.train_metrics.compute()

    def _validate(self):
        """
        Evaluate model on validation set.

        Performs:
        - Model evaluation mode
        - Forward pass without gradients
        - Metrics computation

        Returns:
            Dict[str, float]: Computed validation metrics
        """
        # Set model to evaluation mode (disables dropout, freezes batch norm, etc.)
        self.model.eval()  # Set model to evaluation mode
        self.validate_metrics.reset()  # Reset metrics

        with torch.no_grad():  # No gradient computation needed for validation
            for data, labels in self.dataset.validate_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                self.validate_metrics.update(outputs, labels)

        metrics = self.validate_metrics.compute()
        return metrics

    def test(self):
        """
        Evaluate model on test set.

        Similar to validation but used for final model evaluation.
        Should only be called after training is complete.

        Returns:
            Dict[str, float]: Computed test metrics
        """
        # Set model to evaluation mode (disables dropout, freezes batch norm, etc.)
        self.model.eval()
        self.test_metrics.reset()

        with torch.no_grad():
            for data, labels in self.dataset.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                self.test_metrics.update(outputs, labels)
        metrics = self.test_metrics.compute()
        return metrics


if __name__ == "__main__":
    train = Train(
        hyperparameters=Hyperparameters(batch_size=32, num_epochs=50),
        model=Model(),
    )
    train.train()
