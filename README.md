# 🔢 Handwritten Digit Recognizer

A complete **Handwritten Digit Recognition System** using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Built with Python, TensorFlow/Keras, and Streamlit.

---

## 📋 Project Overview

This project implements an end-to-end machine learning pipeline for recognizing handwritten digits (0–9). It includes data preprocessing, model training, evaluation, and a web-based user interface where users can draw or upload digits for real-time prediction.

## 🎯 Problem Statement

Handwritten digit recognition is a fundamental problem in computer vision and pattern recognition. The goal is to build a system that can accurately classify images of handwritten digits into one of ten categories (0 through 9). This has practical applications in postal mail sorting, bank check processing, and form digitization.

## 🎯 Objectives

- Train a CNN model to classify handwritten digits with **>98% accuracy**
- Implement a complete ML pipeline from data loading to deployment
- Create an intuitive web interface for live digit recognition
- Provide comprehensive model evaluation with visualizations
- Build a clean, modular, and well-documented codebase

## ✨ Features

- **CNN Model**: Custom Convolutional Neural Network optimized for digit recognition
- **High Accuracy**: Achieves **99%+** accuracy on the MNIST test set
- **Web Interface**: Streamlit-based UI with canvas drawing and image upload
- **Real-time Prediction**: Instant digit recognition with confidence scores
- **Probability Distribution**: Visual bar chart showing probabilities for all 10 digits
- **Comprehensive Evaluation**: Confusion matrix, classification report, and misclassification analysis
- **Training Visualization**: Accuracy and loss curves over training epochs
- **Modular Architecture**: Clean separation of concerns for easy understanding

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3 | Core programming language |
| TensorFlow / Keras | CNN model building and training |
| NumPy | Numerical operations |
| Pillow (PIL) | Image loading and preprocessing |
| Matplotlib | Training plots and visualizations |
| Seaborn | Confusion matrix heatmap |
| Scikit-learn | Evaluation metrics |
| Streamlit | Web application framework |
| streamlit-drawable-canvas | Canvas drawing component |

## 📊 Dataset

**MNIST (Modified National Institute of Standards and Technology)**

- **Training set**: 60,000 images
- **Test set**: 10,000 images
- **Image size**: 28 × 28 pixels (grayscale)
- **Classes**: 10 (digits 0–9)
- **Source**: Automatically downloaded via `keras.datasets.mnist`

The dataset is balanced, with roughly equal representation of each digit class.

## 🏗️ CNN Architecture

```
Input (28×28×1)
  → Conv2D(32 filters, 3×3, ReLU, padding='same')    → 28×28×32
  → MaxPooling2D(2×2)                                  → 14×14×32
  → Conv2D(64 filters, 3×3, ReLU, padding='same')     → 14×14×64
  → MaxPooling2D(2×2)                                  → 7×7×64
  → Flatten()                                           → 3,136
  → Dense(128, ReLU)                                    → 128
  → Dropout(0.5)
  → Dense(10, Softmax)                                  → 10 probabilities
```

**Total Parameters**: ~421,642

### Architecture Design Rationale

| Component | Choice | Why |
|-----------|--------|-----|
| Conv layers | 2 blocks | Sufficient for 28×28 images; deeper networks are overkill |
| Filters | 32 → 64 | Progressive increase captures low → high-level features |
| Kernel size | 3×3 | Small kernels capture fine details efficiently |
| Pooling | 2×2 MaxPool | Reduces dimensions, adds translation invariance |
| Dense units | 128 | Enough capacity without overfitting on MNIST |
| Dropout | 0.5 | Strong regularization for the dense layer |
| Activation | ReLU + Softmax | ReLU avoids vanishing gradients; Softmax gives probabilities |
| Optimizer | Adam (lr=0.001) | Adaptive learning rate, fast convergence |

## 🔄 System Workflow

```
MNIST Dataset
  → Data Loading (Keras)
  → Data Exploration (sample visualization, class distribution)
  → Image Preprocessing (normalize, reshape, one-hot encode)
  → Train/Test Split (60k train / 10k test)
  → CNN Architecture (build and compile)
  → Model Training (with early stopping & checkpointing)
  → Validation (monitor val_loss and val_accuracy)
  → Evaluation (confusion matrix, classification report)
  → Model Saving (digit_cnn.keras)
  → Prediction (preprocess → CNN → probability distribution)
  → Streamlit Deployment (web UI)
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏋️ How to Train the Model

Run the training script from the project root directory:

```bash
python -m src.train
```

This will:
1. Download the MNIST dataset (first run only, ~11 MB)
2. Display dataset information and sample images
3. Preprocess the data (normalize, reshape, encode)
4. Build the CNN model
5. Train for up to 15 epochs with early stopping
6. Save the best model to `models/digit_cnn.keras`
7. Generate training history plots in `plots/`

**Expected training time**: ~2–3 minutes on CPU

### Evaluate the Model

After training, run the evaluation script:

```bash
python -m src.evaluate
```

This generates:
- Confusion matrix heatmap
- Per-class classification report
- Correctly classified examples
- Misclassified examples with analysis

## 🖥️ How to Run the Streamlit Application

```bash
streamlit run app/app.py
```

The application will open in your browser at `http://localhost:8501`.

### Features:
- **Home**: Project overview and model information
- **Recognize Digit**: Draw on canvas or upload an image to get predictions
- **About Model**: CNN architecture details and evaluation results

## 📈 Expected Results

| Metric | Expected Value |
|--------|---------------|
| Test Accuracy | >99% |
| Test Loss | <0.05 |
| Weighted Precision | >99% |
| Weighted Recall | >99% |
| Weighted F1-Score | >99% |

## 📊 Evaluation Metrics

- **Accuracy**: Percentage of correctly classified digits
- **Precision**: Of all digits predicted as class X, how many were actually X?
- **Recall**: Of all actual class X digits, how many were correctly identified?
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Shows which digits are commonly confused with each other


### Training History
*Training and validation accuracy/loss curves over epochs*

### Confusion Matrix
*Heatmap showing prediction distribution across digit classes*

### Web Application
*Streamlit interface with canvas drawing and prediction results*

## 🔮 Future Improvements

- [ ] Add data augmentation (rotation, scaling, shifting) for better generalization
- [ ] Experiment with deeper architectures (e.g., adding BatchNormalization)
- [ ] Add support for multi-digit recognition
- [ ] Implement real-time webcam digit recognition
- [ ] Deploy to cloud (Streamlit Cloud, Heroku, or AWS)
- [ ] Add model comparison (CNN vs. SVM vs. Random Forest)
- [ ] Implement Grad-CAM visualization to show what the CNN "sees"

## ⚠️ Limitations

- **Single digit only**: The system recognizes one digit at a time
- **MNIST bias**: Trained on centered, size-normalized digits — may struggle with:
  - Very thick or thin strokes
  - Digits written at extreme angles
  - Non-standard writing styles
- **No multi-digit support**: Cannot recognize numbers like "42" or "100"
- **Black background expected**: Works best when digits are white/light on dark background
- **28×28 resolution**: All images are downscaled, which may lose detail from high-res inputs

## 📝 Conclusion

This project demonstrates a complete machine learning workflow from data preprocessing to web deployment. The CNN model achieves excellent accuracy on the MNIST dataset, making it suitable for academic demonstration and as a foundation for more complex handwriting recognition systems.

The modular code structure, comprehensive evaluation, and user-friendly interface make this project ideal for BTech students looking to understand:
- How CNNs process image data
- The importance of data preprocessing
- Model evaluation best practices
- Web deployment of ML models

---

## 📁 Project Structure

```
handwritten-digit-recognizer/
│
├── data/                      # MNIST dataset cache (auto-created)
│
├── models/                    # Saved trained models
│   └── digit_cnn.keras
│
├── plots/                     # Generated evaluation plots
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── correct_predictions.png
│   ├── incorrect_predictions.png
│   ├── sample_images.png
│   └── class_distribution.png
│
├── src/                       # Source code modules
│   ├── __init__.py
│   ├── config.py              # Hyperparameters and paths
│   ├── preprocessing.py       # Data loading and preprocessing
│   ├── model.py               # CNN architecture
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Model evaluation
│   └── predict.py             # Single image prediction
│
├── app/                       # Streamlit web application
│   └── app.py
│
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

---

*Built with ❤️ using TensorFlow and Streamlit*
