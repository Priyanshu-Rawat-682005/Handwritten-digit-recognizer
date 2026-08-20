"""
evaluate.py — Model Evaluation and Performance Analysis

Run after training:  python -m src.evaluate
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

from src.config import MODEL_PATH, PLOTS_DIR, NUM_CLASSES, create_directories
from src.preprocessing import load_mnist_data, preprocess_data


def load_trained_model():
    """Load the trained CNN model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at '{MODEL_PATH}'.\n"
            f"Please train first: python -m src.train"
        )
    print(f"  Loading model from {MODEL_PATH}...")
    model = keras.models.load_model(MODEL_PATH)
    print("  [OK] Model loaded successfully")
    return model


def evaluate_model(model, x_test, y_test):
    """Evaluate the model and print metrics."""
    print("\n" + "=" * 60)
    print("  Model Evaluation on Test Set")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n  Test Accuracy:  {test_accuracy * 100:.2f}%")
    print(f"  Test Loss:      {test_loss:.4f}")

    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    precision = precision_score(y_true_classes, y_pred_classes, average="weighted")
    recall = recall_score(y_true_classes, y_pred_classes, average="weighted")
    f1 = f1_score(y_true_classes, y_pred_classes, average="weighted")

    print(f"\n  Weighted Precision: {precision * 100:.2f}%")
    print(f"  Weighted Recall:    {recall * 100:.2f}%")
    print(f"  Weighted F1-Score:  {f1 * 100:.2f}%")

    return test_loss, test_accuracy, y_pred_classes, y_true_classes


def plot_confusion_matrix(y_true, y_pred, save=True):
    """Generate a confusion matrix heatmap."""
    create_directories()
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(NUM_CLASSES),
                yticklabels=range(NUM_CLASSES),
                square=True, linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted Digit", fontsize=13)
    ax.set_ylabel("True Digit", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save:
        save_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  [OK] Confusion matrix saved to {save_path}")
    plt.close()


def print_classification_report(y_true, y_pred):
    """Print per-class precision, recall, and F1-score."""
    print("\n" + "=" * 60)
    print("  Detailed Classification Report")
    print("=" * 60)
    digit_names = [f"Digit {i}" for i in range(NUM_CLASSES)]
    report = classification_report(y_true, y_pred,
                                   target_names=digit_names, digits=4)
    print(f"\n{report}")


def plot_correct_predictions(x_test, y_true, y_pred, num_examples=9, save=True):
    """Display examples of correctly classified digits."""
    create_directories()
    correct_idx = np.where(y_true == y_pred)[0]
    selected = np.random.choice(correct_idx, min(num_examples, len(correct_idx)),
                                replace=False)
    rows = cols = int(np.ceil(np.sqrt(num_examples)))

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    fig.suptitle("Correctly Classified Digits", fontsize=16,
                 fontweight="bold", color="green")
    for i, ax in enumerate(axes.flat):
        if i < len(selected):
            idx = selected[i]
            ax.imshow(x_test[idx].squeeze(), cmap="gray")
            ax.set_title(f"True: {y_true[idx]} | Pred: {y_pred[idx]}",
                         fontsize=10, color="green")
        ax.axis("off")
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "correct_predictions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Correct predictions saved to {path}")
    plt.close()


def plot_incorrect_predictions(x_test, y_true, y_pred, num_examples=9, save=True):
    """Display misclassified digits with analysis."""
    create_directories()
    incorrect_idx = np.where(y_true != y_pred)[0]
    if len(incorrect_idx) == 0:
        print("  No misclassifications found!")
        return

    selected = np.random.choice(incorrect_idx,
                                min(num_examples, len(incorrect_idx)),
                                replace=False)
    rows = cols = int(np.ceil(np.sqrt(num_examples)))

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    fig.suptitle("Misclassified Digits", fontsize=16,
                 fontweight="bold", color="red")
    for i, ax in enumerate(axes.flat):
        if i < len(selected):
            idx = selected[i]
            ax.imshow(x_test[idx].squeeze(), cmap="gray")
            ax.set_title(f"True: {y_true[idx]} | Pred: {y_pred[idx]}",
                         fontsize=10, color="red")
        ax.axis("off")
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "incorrect_predictions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [OK] Incorrect predictions saved to {path}")
    plt.close()

    print("\n  Common misclassification reasons:")
    print("  - 4 vs 9 : Similar vertical strokes")
    print("  - 3 vs 8 : Similar curved strokes")
    print("  - 7 vs 1 : Without cross-stroke, 7 resembles 1")
    print("  - 5 vs 3 : Similar lower curves")


def run_evaluation():
    """Execute the complete evaluation pipeline."""
    create_directories()
    print("\n" + "=" * 60)
    print("  HANDWRITTEN DIGIT RECOGNIZER - EVALUATION")
    print("=" * 60)

    model = load_trained_model()
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    x_train, y_train, x_test, y_test = preprocess_data(
        x_train, y_train, x_test, y_test
    )
    test_loss, test_accuracy, y_pred, y_true = evaluate_model(model, x_test, y_test)
    plot_confusion_matrix(y_true, y_pred)
    print_classification_report(y_true, y_pred)
    plot_correct_predictions(x_test, y_true, y_pred)
    plot_incorrect_predictions(x_test, y_true, y_pred)

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE!")
    print(f"  All plots saved to: {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
