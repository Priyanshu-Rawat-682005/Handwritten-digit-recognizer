"""
train.py — Model Training Pipeline

This module handles the complete training workflow:
1. Load and preprocess the MNIST dataset
2. Build the CNN model
3. Set up training callbacks (early stopping, checkpointing)
4. Train the model
5. Save training history plots
6. Save the final model

Run this script to train the model:
    python -m src.train
"""

import os
import matplotlib.pyplot as plt
from tensorflow import keras

from src.config import (
    BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENCE,
    MODEL_PATH, MODEL_DIR, PLOTS_DIR,
    create_directories
)
from src.preprocessing import (
    load_mnist_data, preprocess_data,
    display_sample_images, display_class_distribution
)
from src.model import build_cnn_model, print_model_summary


def get_callbacks():
    """
    Create training callbacks for monitoring and optimization.

    Callbacks are functions that are called at certain points during training.
    They allow us to:
    - Stop training early when the model stops improving
    - Save the best version of the model automatically

    Returns:
        list: List of Keras callback objects
    """
    callbacks = []

    # Early Stopping: Monitor validation loss and stop training if it
    # doesn't improve for 'patience' consecutive epochs.
    # restore_best_weights=True ensures we keep the best model, not the last one.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)

    # Model Checkpoint: Save the model whenever validation loss improves.
    # save_best_only=True means we only keep the single best model,
    # not a checkpoint for every epoch.
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    callbacks.append(model_checkpoint)

    return callbacks


def plot_training_history(history):
    """
    Generate and save plots showing how training progressed.

    Two plots are created:
    1. Accuracy over epochs (training vs. validation)
    2. Loss over epochs (training vs. validation)

    These plots help us understand:
    - Whether the model is learning (accuracy increasing, loss decreasing)
    - Whether the model is overfitting (training accuracy >> validation accuracy)
    - When the model converged (curves flatten out)

    Args:
        history: Keras History object returned by model.fit()
    """
    create_directories()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ─── Accuracy Plot ───────────────────────────────────────────────
    ax1.plot(history.history["accuracy"], label="Training Accuracy",
             color="#2196F3", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy",
             color="#FF9800", linewidth=2, linestyle="--")
    ax1.set_title("Model Accuracy Over Epochs", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.95, 1.0])  # MNIST accuracy is typically 98-99%+

    # ─── Loss Plot ───────────────────────────────────────────────────
    ax2.plot(history.history["loss"], label="Training Loss",
             color="#2196F3", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Validation Loss",
             color="#FF9800", linewidth=2, linestyle="--")
    ax2.set_title("Model Loss Over Epochs", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, "training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  [OK] Training history plots saved to {save_path}")
    plt.close()


def train_model():
    """
    Execute the complete training pipeline.

    This function orchestrates the entire training process:
    1. Creates necessary directories
    2. Loads and preprocesses data
    3. Displays sample images and class distribution
    4. Builds and compiles the CNN
    5. Trains with callbacks
    6. Plots and saves results

    Returns:
        tuple: (model, history) — trained model and training history
    """
    # Create project directories
    create_directories()

    print("\n" + "=" * 60)
    print("  HANDWRITTEN DIGIT RECOGNIZER - TRAINING PIPELINE")
    print("=" * 60)

    # ─── Step 1: Load Data ───────────────────────────────────────────
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Display sample images before preprocessing
    display_sample_images(x_train, y_train, title="Raw MNIST Training Samples")
    display_class_distribution(y_train, title="Training Set Class Distribution")

    # ─── Step 2: Preprocess Data ─────────────────────────────────────
    x_train, y_train, x_test, y_test = preprocess_data(
        x_train, y_train, x_test, y_test
    )

    # ─── Step 3: Build Model ─────────────────────────────────────────
    model = build_cnn_model()
    print_model_summary(model)

    # ─── Step 4: Train Model ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training the CNN")
    print("=" * 60)
    print(f"\n  Batch size:    {BATCH_SIZE}")
    print(f"  Max epochs:    {EPOCHS}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Model save path: {MODEL_PATH}")
    print()

    callbacks = get_callbacks()

    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # ─── Step 5: Save Results ────────────────────────────────────────
    plot_training_history(history)

    # Save the final model (in case early stopping didn't trigger a save)
    model.save(MODEL_PATH)
    print(f"\n  [OK] Final model saved to {MODEL_PATH}")

    # ─── Step 6: Quick Evaluation ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Quick Evaluation on Test Set")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n  Test Accuracy:  {test_accuracy * 100:.2f}%")
    print(f"  Test Loss:      {test_loss:.4f}")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)

    return model, history


# ─── Run Training ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_model()
