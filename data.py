"""
This module handles the MNIST dataset loading, preprocessing, and visualization.
It provides functionality for loading the MNIST dataset with data augmentation,
splitting it into train/validation/test sets, and visualizing the samples.
"""

from torchvision import datasets, transforms
import torch
from matplotlib import pyplot as plt
import matplotlib


class DataSet(object):
    """
    A class for handling MNIST dataset operations including loading, preprocessing, and visualization.

    This class provides functionality for:
    - Loading MNIST dataset with data augmentation for training
    - Splitting data into train/validation/test sets
    - Creating DataLoader instances for batch processing
    - Visualizing sample images
    - Random sample selection

    Args:
        root (str, optional): Directory for storing dataset. Defaults to "./data"
        batch_size (int, optional): Size of batches for DataLoaders. Defaults to 10
    """

    def __init__(self, root="./data", batch_size=10):
        """
        Initialize the dataset with data augmentation and splitting.

        The initialization process includes:
        1. Setting up data augmentation for training (rotation, affine transforms)
        2. Loading MNIST dataset for both training and testing
        3. Splitting training data into train and validation sets
        4. Creating DataLoader instances for batch processing

        Args:
            root (str, optional): Directory to store the dataset. Defaults to "./data"
            batch_size (int, optional): Batch size for DataLoaders. Defaults to 10
        """
        # Define data augmentation for training data
        # This includes random rotation, translation, and scaling to improve model robustness
        train_transform = transforms.Compose(
            [
                transforms.RandomRotation(degrees=15),  # Rotate images up to 15 degrees
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.2, 0.2),
                    scale=(0.8, 1.2),  # Apply random translation and scaling
                ),
                transforms.ToTensor(),  # Convert images to PyTorch tensors
            ]
        )
        self.train_dataset = datasets.MNIST(
            root, train=True, transform=train_transform, download=True
        )
        self.test_dataset = datasets.MNIST(
            root, train=False, transform=transforms.ToTensor(), download=True
        )
        # Split training data into training and validation sets
        # Validation set size matches test set size for balanced evaluation
        self.train_dataset, self.validate_dataset = torch.utils.data.random_split(
            self.train_dataset,
            [len(self.train_dataset) - len(self.test_dataset), len(self.test_dataset)],
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size, shuffle=False
        )
        self.validate_loader = torch.utils.data.DataLoader(
            self.validate_dataset, batch_size, shuffle=False
        )

    def show_samples(self, num_samples=10, n_cols=5):
        """
        Display a grid of sample images from the training dataset.

        Creates a figure with multiple subplots showing MNIST digits and their labels.
        If needed, fetches multiple batches to get the requested number of samples.

        Args:
            num_samples (int, optional): Number of images to display. Defaults to 10
            n_cols (int, optional): Number of columns in the grid. Defaults to 5
        """
        images, labels = next(iter(self.train_loader))
        while len(images) < num_samples:
            new_images, new_labels = next(iter(self.train_loader))
            images = torch.cat([images, new_images])
            labels = torch.cat([labels, new_labels])

        matplotlib.use("qt5agg")

        n_rows = (num_samples + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(10, 2 * n_rows))  # 10 for width, 2.5 for height
        for i in range(num_samples):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.imshow(images[i].squeeze(), cmap="gray")
            ax.axis("off")
            ax.set_title(f"Label: {labels[i].item()}")

        plt.tight_layout()
        plt.show()

    def random_one(self):
        """
        Select a random sample from any of the datasets (train/test/validation).

        This method:
        1. Calculates total size of all datasets
        2. Generates a random index
        3. Maps the index to the appropriate dataset
        4. Returns the selected image and its label

        Returns:
            tuple: (image, label) where image is a tensor and label is an integer
        """
        # Calculate sizes of each dataset
        train_size = len(self.train_dataset)
        test_size = len(self.test_dataset)
        val_size = len(self.validate_dataset)
        total_size = train_size + test_size + val_size

        # Generate a random index within the total size of all datasets
        random_idx = torch.randint(0, total_size, (1,)).item()

        # Map the random index to the appropriate dataset and calculate local index
        if random_idx < train_size:
            dataset = self.train_dataset
            idx = random_idx  # Use index directly for training set
        elif random_idx < train_size + test_size:
            dataset = self.test_dataset
            idx = random_idx - train_size  # Adjust index for test set
        else:
            dataset = self.validate_dataset
            idx = random_idx - train_size - test_size  # Adjust index for validation set

        image, label = dataset[int(idx)]
        return image, label


if __name__ == "__main__":
    dataset = DataSet()
    dataset.show_samples()
