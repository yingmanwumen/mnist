"""
This module implements the inference and visualization functionality for the MNIST classifier.
It provides:
- Model inference on random samples
- Interactive visualization of predictions
- Probability distribution display for each digit
- Continuous inference loop with graceful exit handling
"""

import torch
from model import Model
from data import DataSet
import matplotlib
from matplotlib import pyplot as plt
import signal


class Reason(object):
    """
    Class for handling model inference on MNIST digits.

    This class provides functionality to:
    - Load a trained model
    - Perform inference on random samples
    - Return predictions along with input data and true labels

    Args:
        dataset (DataSet): Dataset instance for sampling images
        model (Model): Trained neural network model
    """

    def __init__(self, dataset: DataSet, model: Model):
        """
        Initialize the reasoning environment.

        Args:
            dataset (DataSet): Dataset instance for sampling images
            model (Model): Trained neural network model
        """
        self.model = model
        self.dataset = dataset

    def reason(self):
        """
        Perform inference on a random sample.

        Returns:
            tuple: (
                data: Input image tensor,
                output: Model predictions (logits),
                label: True label
            )
        """
        with torch.no_grad():
            self.model.eval()  # Set model to evaluation mode
            data, label = self.dataset.random_one()
            return data, self.model(data.unsqueeze(0)), label


def display(data, output, target):
    """
    Display the input image and model predictions.

    Creates a figure showing:
    - The input MNIST digit image
    - The true label (target)
    - The model's prediction
    - Probability distribution for all digits (0-9)

    Args:
        data (torch.Tensor): Input image tensor
        output (torch.Tensor): Model output logits
        target (int): True label for the image
    """
    # Set matplotlib backend for interactive display
    matplotlib.use("qt5agg")

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(data.squeeze(), cmap="gray")
    ax.axis("off")
    # Get the predicted digit (class with highest probability)
    result = output.argmax(1)
    # Convert logits to probabilities using softmax
    output = torch.nn.functional.softmax(output, dim=1)[0]
    title = f"Target: {target}\n"
    title += f"Result: {result.item()}\n"
    for i, x in enumerate(output):
        title += f"{i}: {x:.4f}\n"
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load the best trained model and set up inference
    # Run continuous inference loop until interrupted
    model_path = "./best_model.pth"
    model = Model()
    model.load_state_dict(torch.load(model_path))

    reason = Reason(DataSet(), model)

    # Set up graceful exit on Ctrl+C
    signal.signal(signal.SIGINT, lambda *_: exit(0))

    # Continuous inference loop
    # Press Ctrl+C to exit
    while True:
        data, output, target = reason.reason()
        display(data, output, target)
