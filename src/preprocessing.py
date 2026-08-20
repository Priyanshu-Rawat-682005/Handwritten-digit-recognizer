"""
preprocessing.py — Data Loading and Preprocessing for MNIST

This module handles:
1. Loading the MNIST dataset from Keras
2. Normalizing pixel values from [0, 255] to [0, 1]
3. Reshaping images to include the channel dimension
4. One-hot encoding the labels
5. Displaying sample images for visual verification

WHY NORMALIZE?
Neural networks work best when input values are small and centered.
Raw pixel values (0-255) can cause:
- Large weight updates that destabilize training
- Slower convergence
- Numerical overflow issues
Scaling to [0, 1] gives the optimizer a smoother loss landscape.

WHY RESHAPE?
Conv2D layers in Keras expect a 4D tensor: (batch_size, height, width, channels).
MNIST images are stored as 2D arrays (28, 28), so we add the channel dimension
to get (28, 28, 1) — the 1 represents a single grayscale channel.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from src.config import (
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
    NUM_CLASSES, PLOTS_DIR, create_directories
)


def load_mnist_data():
    """
    Load the MNIST dataset from Keras.

    MNIST contains:
    - 60,000 training images of handwritten digits (0-9)
    - 10,000 test images for evaluation
    - Each image is 28x28 pixels in grayscale

    Returns:
        tuple: (x_train, y_train), (x_test, y_test)
            - x: numpy arrays of pixel values (uint8, 0-255)
            - y: numpy arrays of integer labels (0-9)
    """
    print("=" * 60)
    print("  Loading MNIST Dataset")
    print("=" * 60)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Display dataset information
    print(f"\n  Training set:  {x_train.shape[0]:,} images")
    print(f"  Test set:      {x_test.shape[0]:,} images")
    print(f"  Image size:    {x_train.shape[1]}x{x_train.shape[2]} pixels")
    print(f"  Pixel range:   [{x_train.min()}, {x_train.max()}]")
    print(f"  Label range:   [{y_train.min()}, {y_train.max()}]")
    print(f"  Data type:     {x_train.dtype}")

    return (x_train, y_train), (x_test, y_test)


def preprocess_data(x_train, y_train, x_test, y_test):
    """
    Preprocess the MNIST data for CNN training.

    Steps:
    1. Normalize pixel values from [0, 255] to [0, 1]
    2. Reshape images from (28, 28) to (28, 28, 1)
    3. One-hot encode labels from integers to binary vectors

    Args:
        x_train: Training images (N, 28, 28), uint8
        y_train: Training labels (N,), integers 0-9
        x_test:  Test images (M, 28, 28), uint8
        y_test:  Test labels (M,), integers 0-9

    Returns:
        tuple: (x_train, y_train, x_test, y_test) — preprocessed
    """
    print("\n" + "=" * 60)
    print("  Preprocessing Data")
    print("=" * 60)

    # Step 1: Normalize pixel values to [0, 1]
    # Dividing by 255.0 scales values from [0, 255] to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    print(f"\n  [OK] Normalized pixel values to [{x_train.min()}, {x_train.max()}]")

    # Step 2: Reshape to add channel dimension
    # From (N, 28, 28) → (N, 28, 28, 1)
    x_train = x_train.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    x_test = x_test.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    print(f"  [OK] Reshaped images to {x_train.shape[1:]}")

    # Step 3: One-hot encode labels
    # Example: label 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # This is required because the output layer has 10 neurons with softmax
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    print(f"  [OK] One-hot encoded labels to {y_train.shape[1]} classes")

    # Verify final dimensions
    print(f"\n  Final shapes:")
    print(f"    x_train: {x_train.shape}")
    print(f"    y_train: {y_train.shape}")
    print(f"    x_test:  {x_test.shape}")
    print(f"    y_test:  {y_test.shape}")

    # Assertions to catch any issues early
    assert x_train.shape[1:] == (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), \
        f"Unexpected training image shape: {x_train.shape[1:]}"
    assert y_train.shape[1] == NUM_CLASSES, \
        f"Unexpected label shape: {y_train.shape[1]}"

    return x_train, y_train, x_test, y_test


def display_sample_images(x_data, y_data, num_rows=5, num_cols=5,
                          title="Sample MNIST Images", save=True):
    """
    Display a grid of sample images from the dataset.

    This is useful for visual verification before training — confirming
    that the data looks correct and labels match the images.

    Args:
        x_data:   Image array (N, 28, 28) or (N, 28, 28, 1)
        y_data:   Labels — integer array or one-hot encoded
        num_rows: Number of rows in the grid
        num_cols: Number of columns in the grid
        title:    Title for the plot
        save:     Whether to save the plot to the plots directory
    """
    create_directories()

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i < len(x_data):
            # Handle both (28,28) and (28,28,1) shapes
            image = x_data[i].squeeze()

            # Handle both integer and one-hot encoded labels
            if len(y_data.shape) > 1:
                label = np.argmax(y_data[i])
            else:
                label = y_data[i]

            ax.imshow(image, cmap="gray")
            ax.set_title(f"Label: {label}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()

    if save:
        save_path = f"{PLOTS_DIR}/sample_images.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  [OK] Sample images saved to {save_path}")

    plt.close()


def display_class_distribution(y_data, title="Class Distribution", save=True):
    """
    Display the distribution of digit classes in the dataset.

    A balanced dataset (roughly equal samples per class) is important
    because it prevents the model from being biased toward any particular digit.

    Args:
        y_data: Labels — integer array or one-hot encoded
        title:  Title for the plot
        save:   Whether to save the plot
    """
    create_directories()

    # Convert one-hot to integer labels if necessary
    if len(y_data.shape) > 1:
        labels = np.argmax(y_data, axis=1)
    else:
        labels = y_data

    # Count occurrences of each digit
    unique, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(unique, counts, color="#4A90D9", edgecolor="white", linewidth=0.5)

    # Add count labels on top of each bar
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{count:,}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Digit", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(10))
    plt.tight_layout()

    if save:
        save_path = f"{PLOTS_DIR}/class_distribution.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Class distribution saved to {save_path}")

    plt.close()
