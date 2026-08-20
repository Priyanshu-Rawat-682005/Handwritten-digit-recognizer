"""
model.py — CNN Architecture for Handwritten Digit Recognition

This module defines the Convolutional Neural Network used to classify
handwritten digits from the MNIST dataset.

ARCHITECTURE OVERVIEW:
    Input (28x28x1)
      → Conv2D(32 filters, 3x3, ReLU, padding='same')   → 28x28x32
      → MaxPooling2D(2x2)                                 → 14x14x32
      → Conv2D(64 filters, 3x3, ReLU, padding='same')    → 14x14x64
      → MaxPooling2D(2x2)                                 → 7x7x64
      → Flatten()                                          → 3136
      → Dense(128, ReLU)                                   → 128
      → Dropout(0.5)
      → Dense(10, Softmax)                                 → 10 (probabilities)

WHY THIS ARCHITECTURE WORKS WELL FOR MNIST:

1. TWO CONV BLOCKS: MNIST images are small (28x28), so deep architectures
   like VGG or ResNet are overkill. Two convolutional blocks provide enough
   feature extraction power to capture edges (block 1) and shapes (block 2).

2. 32 → 64 FILTER PROGRESSION: As spatial dimensions shrink through pooling,
   we increase filter count to capture more complex features. This is a
   standard and effective pattern in CNN design.

3. 3x3 KERNELS: Small kernels are computationally efficient and can capture
   fine-grained features. Multiple small kernels stacked together (via depth)
   can approximate larger receptive fields.

4. padding='same': Preserves spatial dimensions before pooling, ensuring no
   information is lost at the borders of the image.

5. ReLU ACTIVATION: Introduces non-linearity, allows the network to learn
   complex patterns. ReLU is computationally efficient and avoids the
   vanishing gradient problem common with sigmoid/tanh.

6. MAXPOOLING: Reduces spatial dimensions by half, making the network:
   - More robust to small translations in the input
   - Computationally more efficient
   - Less prone to overfitting

7. DROPOUT (0.5): Randomly disables 50% of neurons in the dense layer during
   training. This prevents co-adaptation of features and acts as strong
   regularization. Particularly important because the Dense(128) layer has
   the most parameters in the network.

8. SOFTMAX OUTPUT: Converts raw scores into a probability distribution over
   the 10 digit classes, where all probabilities sum to 1. This gives us
   both the predicted class and a confidence score.
"""

from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    INPUT_SHAPE, NUM_CLASSES,
    CONV1_FILTERS, CONV2_FILTERS, KERNEL_SIZE, POOL_SIZE,
    DENSE_UNITS, DROPOUT_RATE
)


def build_cnn_model():
    """
    Build and compile the CNN model for digit recognition.

    The model uses the Sequential API for simplicity and readability.
    Each layer is added in order, making the data flow easy to understand.

    Returns:
        keras.Model: Compiled CNN model ready for training
    """
    model = keras.Sequential([
        # ─── First Convolutional Block ───────────────────────────────
        # Input: 28x28x1 → Output: 14x14x32

        # Conv2D: Applies 32 learnable 3x3 filters to the input image.
        # Each filter detects a specific feature (edges, corners, curves).
        # ReLU activation: max(0, x) — introduces non-linearity.
        # padding='same' keeps output size = input size before pooling.
        layers.Conv2D(
            filters=CONV1_FILTERS,
            kernel_size=KERNEL_SIZE,
            activation="relu",
            padding="same",
            input_shape=INPUT_SHAPE,
            name="conv2d_block1"
        ),

        # MaxPooling: Takes the maximum value in each 2x2 window.
        # Reduces spatial dimensions from 28x28 to 14x14.
        layers.MaxPooling2D(
            pool_size=POOL_SIZE,
            name="maxpool_block1"
        ),

        # ─── Second Convolutional Block ──────────────────────────────
        # Input: 14x14x32 → Output: 7x7x64

        # Conv2D: 64 filters capture more complex, higher-level features
        # like parts of digits (loops, strokes, intersections).
        layers.Conv2D(
            filters=CONV2_FILTERS,
            kernel_size=KERNEL_SIZE,
            activation="relu",
            padding="same",
            name="conv2d_block2"
        ),

        # MaxPooling: Reduces from 14x14 to 7x7.
        layers.MaxPooling2D(
            pool_size=POOL_SIZE,
            name="maxpool_block2"
        ),

        # ─── Classifier Head ────────────────────────────────────────
        # Input: 7x7x64 = 3136 features → Output: 10 class probabilities

        # Flatten: Converts the 3D feature maps (7x7x64) into a 1D vector
        # of 3136 values. This is required before passing to Dense layers.
        layers.Flatten(name="flatten"),

        # Dense: Fully connected layer with 128 neurons.
        # Learns to combine the extracted features for classification.
        layers.Dense(
            DENSE_UNITS,
            activation="relu",
            name="dense_hidden"
        ),

        # Dropout: Randomly sets 50% of inputs to 0 during training.
        # This prevents the network from memorizing the training data
        # and forces it to learn more generalizable features.
        layers.Dropout(
            DROPOUT_RATE,
            name="dropout"
        ),

        # Output: 10 neurons, one per digit class (0-9).
        # Softmax converts raw scores to probabilities that sum to 1.
        # The highest probability corresponds to the predicted digit.
        layers.Dense(
            NUM_CLASSES,
            activation="softmax",
            name="output"
        ),
    ])

    # Compile the model with optimizer, loss function, and metrics
    model.compile(
        # Adam optimizer: Adaptive learning rate, fast and reliable
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        # Categorical crossentropy: Standard loss for multi-class classification
        # with one-hot encoded labels
        loss="categorical_crossentropy",
        # Track accuracy during training
        metrics=["accuracy"]
    )

    return model


def print_model_summary(model):
    """
    Print a detailed summary of the model architecture.

    Shows each layer's name, output shape, and number of parameters.
    Useful for verifying the architecture before training.

    Args:
        model: Compiled Keras model
    """
    print("\n" + "=" * 60)
    print("  CNN Model Architecture")
    print("=" * 60)
    model.summary()

    # Calculate and display total parameters
    total_params = model.count_params()
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Model is {'small' if total_params < 1_000_000 else 'large'} "
          f"- suitable for {'CPU' if total_params < 1_000_000 else 'GPU'} training")
