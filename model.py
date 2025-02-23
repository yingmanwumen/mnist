"""
This module defines a Convolutional Neural Network (CNN) architecture for MNIST digit classification.
The model uses a modern CNN architecture with:
- Multiple convolutional layers with batch normalization
- Max pooling for spatial dimension reduction
- Dropout for regularization
- Fully connected layers for final classification
"""

import torch.nn as nn


class Model(nn.Module):
    """
    A CNN model for MNIST digit classification.

    Architecture:
    1. Convolutional layers:
       - 3 conv layers with increasing channels (32->64->128)
       - Each followed by batch normalization and ReLU
       - MaxPool2D and dropout for downsampling and regularization
    2. Fully connected layers:
       - Flattened input -> 128 features
       - ReLU activation and dropout
    3. Classifier:
       - 128 features -> 10 classes (digits 0-9)

    Input shape: (batch_size, 1, 28, 28)
    Output shape: (batch_size, 10)
    """
    def __init__(self):
        """
        Initialize the model architecture.

        Sets up three main components:
        1. Convolutional layer sequence
        2. Fully connected layer sequence
        3. Final classifier layer
        """
        super(Model, self).__init__()
        # Convolutional layers with batch normalization and pooling
        # Progressively increases feature channels while reducing spatial dimensions
        self.conv_layer = nn.Sequential(
            # 28, 28, 1 -> 28, 28, 32
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 28, 28, 32 -> 28, 28, 64
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 28, 28, 64 -> 14, 14, 128
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 14, 14, 128 -> 7, 7, 128
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        # Fully connected layers for feature extraction
        # Flattens convolutional features and reduces dimensionality
        self.fc_layer = nn.Sequential(
            nn.Flatten(), nn.Linear(7 * 7 * 128, 128), nn.ReLU(), nn.Dropout(0.5)
        )
        # Final classification layer
        # Maps 128 features to 10 digit classes
        self.classifier = nn.Sequential(nn.Linear(128, 10))
        self.print()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            torch.Tensor: Logits for each digit class, shape (batch_size, 10)
        """
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        x = self.classifier(x)
        return x

    def print(self):
        """
        Print the model architecture.
        Useful for debugging and verification of the model structure.
        """
        print(self)
