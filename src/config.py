"""
config.py — Centralized Configuration for the Handwritten Digit Recognizer

All hyperparameters, file paths, and constants are defined here.
This makes it easy to experiment with different settings without
modifying multiple files.
"""

import os

# ─── Project Paths ───────────────────────────────────────────────────────────

# Get the root directory of the project (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory to store trained models
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Path to save/load the best trained model
MODEL_PATH = os.path.join(MODEL_DIR, "digit_cnn.keras")

# Directory to store generated plots
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# Directory for cached dataset files
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


# ─── Image Configuration ────────────────────────────────────────────────────

# MNIST images are 28x28 pixels, single channel (grayscale)
IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHANNELS = 1  # Grayscale = 1 channel

# Full input shape for the CNN (height, width, channels)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Number of output classes (digits 0 through 9)
NUM_CLASSES = 10


# ─── CNN Architecture Hyperparameters ────────────────────────────────────────

# First convolutional layer: 32 filters
# - Captures low-level features like edges, corners, and simple curves
# - 32 is a standard starting point that balances capacity with computation
CONV1_FILTERS = 32

# Second convolutional layer: 64 filters
# - Captures higher-level features like digit parts and shapes
# - Doubling filters is a common practice as spatial dimensions shrink
CONV2_FILTERS = 64

# Kernel size for all convolutional layers: 3x3
# - Small enough to capture fine details in 28x28 images
# - Proven effective across many image classification tasks
KERNEL_SIZE = (3, 3)

# Max pooling size: 2x2
# - Halves the spatial dimensions after each conv block
# - Provides translation invariance and reduces computation
POOL_SIZE = (2, 2)

# Dense layer units: 128
# - Provides enough capacity to learn digit classifications
# - Not too large to cause overfitting on the relatively simple MNIST dataset
DENSE_UNITS = 128

# Dropout rate: 0.5 (50%)
# - Applied after the dense layer to prevent overfitting
# - 0.5 is the most commonly used dropout rate and works well in practice
# - Randomly disables 50% of neurons during training, forcing the network
#   to learn more robust features
DROPOUT_RATE = 0.5


# ─── Training Hyperparameters ───────────────────────────────────────────────

# Optimizer: Adam (Adaptive Moment Estimation)
# - Combines benefits of RMSprop and SGD with momentum
# - Adapts the learning rate for each parameter individually
# - Typically converges faster than plain SGD
OPTIMIZER = "adam"

# Learning rate: 0.001 (Adam's default)
# - Works well for most problems without tuning
# - Adam's adaptive nature makes it less sensitive to the initial LR
LEARNING_RATE = 0.001

# Batch size: 128
# - Good balance between training speed and gradient stability
# - Larger batches are faster but may generalize slightly worse
# - 128 is a practical choice that fits comfortably in memory
BATCH_SIZE = 128

# Maximum number of training epochs: 15
# - MNIST typically converges within 10-15 epochs
# - Early stopping callback will halt training sooner if validation
#   loss stops improving, so this is an upper limit
EPOCHS = 15

# Early stopping patience: 3
# - Stop training if validation loss doesn't improve for 3 consecutive epochs
# - Prevents overfitting and saves training time
EARLY_STOPPING_PATIENCE = 3


# ─── Utility: Create Directories ────────────────────────────────────────────

def create_directories():
    """Create all required project directories if they don't exist."""
    for directory in [MODEL_DIR, PLOTS_DIR, DATA_DIR]:
        os.makedirs(directory, exist_ok=True)
