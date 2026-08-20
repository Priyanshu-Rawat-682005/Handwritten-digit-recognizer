"""
predict.py — Single Image Prediction

This module provides functions to predict a handwritten digit from:
- An image file path (PNG, JPG, etc.)
- A numpy array (e.g., from a canvas)

The prediction pipeline matches the preprocessing used during training:
1. Convert to grayscale
2. Resize to 28x28
3. Invert colors if needed (MNIST has white digits on black background)
4. Normalize to [0, 1]
5. Reshape to (1, 28, 28, 1)

Run standalone:  python -m src.predict --image path/to/digit.png
"""

import os
import numpy as np
from PIL import Image
from tensorflow import keras

from src.config import (
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
    MODEL_PATH, NUM_CLASSES
)


def load_model():
    """
    Load the trained model for prediction.

    Returns:
        keras.Model: Loaded trained model

    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'.\n"
            f"Train the model first: python -m src.train"
        )
    return keras.models.load_model(MODEL_PATH)


def preprocess_image(image_input):
    """
    Preprocess an image for prediction.

    Handles both file paths and numpy arrays. Applies the same
    transformations used during training to ensure consistency.

    Args:
        image_input: Either a file path (str) or numpy array

    Returns:
        numpy.ndarray: Preprocessed image with shape (1, 28, 28, 1)

    Raises:
        ValueError: If input is invalid
        FileNotFoundError: If image file doesn't exist
    """
    # Handle file path input
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")

        # Open and convert to grayscale
        image = Image.open(image_input).convert("L")
        image = np.array(image)

    # Handle numpy array input
    elif isinstance(image_input, np.ndarray):
        image = image_input.copy()

        # If RGB/RGBA, convert to grayscale using the RGB channels.
        # NOTE: st_canvas returns RGBA where alpha is 255 everywhere
        # (both background and strokes), so we must NOT use the alpha
        # channel. Instead, convert RGB to grayscale.
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA — use only RGB, ignore alpha
                image = np.mean(image[:, :, :3], axis=2).astype(np.uint8)
            elif image.shape[2] == 3:  # RGB
                image = np.mean(image, axis=2).astype(np.uint8)
    else:
        raise ValueError(
            f"Expected file path (str) or numpy array, got {type(image_input)}"
        )

    # MNIST has white digits on black background.
    # If the input image has a light background (mean > 127),
    # invert it so digits are white on black.
    if image.mean() > 127:
        image = 255 - image

    # Center the digit in the frame (like MNIST preprocessing).
    # Find the bounding box of the non-zero pixels and center them.
    image = _center_digit(image)

    # Resize to 28x28
    pil_image = Image.fromarray(image.astype(np.uint8))
    pil_image = pil_image.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
    image = np.array(pil_image).astype("float32")

    # Normalize to [0, 1]
    image = image / 255.0

    # Reshape for the CNN: (1, 28, 28, 1)
    image = image.reshape(1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    return image


def _center_digit(image):
    """
    Center the digit within its frame, similar to MNIST preprocessing.

    Finds the bounding box of non-zero pixels, crops the digit,
    and places it centered in a square frame with padding.

    Args:
        image: 2D numpy array (grayscale)

    Returns:
        numpy.ndarray: Centered image
    """
    # Find rows and columns with non-zero pixels
    rows = np.any(image > 20, axis=1)
    cols = np.any(image > 20, axis=0)

    if not rows.any() or not cols.any():
        # No digit found, return as-is
        return image

    # Get bounding box
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop the digit
    digit = image[rmin:rmax + 1, cmin:cmax + 1]

    # Make it square by padding the shorter dimension
    h, w = digit.shape
    size = max(h, w)

    # Add 20% padding around the digit (MNIST-style)
    pad = max(int(size * 0.2), 4)
    canvas_size = size + 2 * pad

    # Create a black canvas and place the digit in the center
    centered = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    y_offset = (canvas_size - h) // 2
    x_offset = (canvas_size - w) // 2
    centered[y_offset:y_offset + h, x_offset:x_offset + w] = digit

    return centered


def predict_digit(image_input, model=None):
    """
    Predict the digit in an image.

    Args:
        image_input: File path (str) or numpy array of the digit image
        model:       Pre-loaded Keras model (optional, loads if None)

    Returns:
        dict: Prediction results containing:
            - 'digit': Predicted digit (int, 0-9)
            - 'confidence': Confidence percentage (float)
            - 'probabilities': Array of probabilities for each digit
    """
    # Load model if not provided
    if model is None:
        model = load_model()

    # Preprocess the image
    processed_image = preprocess_image(image_input)

    # Get prediction probabilities from the model
    probabilities = model.predict(processed_image, verbose=0)[0]

    # The predicted digit is the class with highest probability
    predicted_digit = int(np.argmax(probabilities))

    # Confidence is the probability of the predicted class
    confidence = float(probabilities[predicted_digit]) * 100

    return {
        "digit": predicted_digit,
        "confidence": confidence,
        "probabilities": probabilities
    }


def format_prediction(result):
    """
    Format prediction results for display.

    Args:
        result: Dictionary from predict_digit()

    Returns:
        str: Formatted prediction string
    """
    output = []
    output.append("=" * 40)
    output.append("  PREDICTION RESULT")
    output.append("=" * 40)
    output.append(f"  Predicted Digit:  {result['digit']}")
    output.append(f"  Confidence:       {result['confidence']:.2f}%")
    output.append("")
    output.append("  Probability Distribution:")
    for i in range(NUM_CLASSES):
        prob = result['probabilities'][i] * 100
        bar = "#" * int(prob / 2)
        output.append(f"    {i}: {prob:6.2f}%  {bar}")
    output.append("=" * 40)
    return "\n".join(output)


# ─── Command-line Usage ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict a handwritten digit")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the digit image file")
    args = parser.parse_args()

    result = predict_digit(args.image)
    print(format_prediction(result))
