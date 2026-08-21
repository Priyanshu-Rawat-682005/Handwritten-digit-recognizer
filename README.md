# Handwritten Digit Recognizer

## About
This is a handwritten digit recognition project I built during my internship to learn the basics of deep learning and computer vision. It uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to classify handwritten numbers from 0 to 9. I also created a Streamlit web app where you can draw a digit on a canvas or upload an image file to get live predictions.

## Features
- Interactive Streamlit web interface with drawing canvas and file upload option.
- Real-time digit prediction with confidence score and 10-digit probability distribution.
- Automated image preprocessing pipeline (grayscale conversion, background inversion, digit centering, and resizing to 28x28).
- 2-block CNN model built using Keras with Convolution, Max Pooling, and Dropout layers.
- Training and evaluation scripts that output training graphs, confusion matrix heatmaps, and prediction samples.

## Technologies Used
- Python 3
- TensorFlow / Keras
- Streamlit & streamlit-drawable-canvas
- NumPy
- OpenCV & Pillow
- Matplotlib & Seaborn
- Scikit-learn

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/handwritten-digit-recognizer.git
   cd handwritten-digit-recognizer
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model (optional, pre-trained model is saved in `models/`):**
   ```bash
   python -m src.train
   ```

5. **Evaluate the model performance:**
   ```bash
   python -m src.evaluate
   ```

6. **Run the Streamlit web application:**
   ```bash
   streamlit run app/app.py
   ```

## How It Works
- **Data Preparation (`src/preprocessing.py`)**: Loads MNIST data (60,000 train / 10,000 test images), normalizes pixels from `[0, 255]` to `[0, 1]`, reshapes to `(28, 28, 1)`, and one-hot encodes labels.
- **Model Architecture (`src/model.py`)**: Built using Keras Sequential API with two Conv2D blocks (32 and 64 filters, 3x3 kernel, ReLU) paired with 2x2 MaxPooling, followed by Flatten, Dense (128 units), 50% Dropout, and a 10-unit Softmax layer.
- **Training Pipeline (`src/train.py`)**: Trains using Adam optimizer and categorical cross-entropy loss. Uses `EarlyStopping` (patience of 3 epochs) and saves the best model to `models/digit_cnn.keras`.
- **Inference (`src/predict.py`)**: Takes drawing or image input, converts RGB/RGBA to grayscale, inverts light backgrounds, crops and centers the digit bounding box with padding, resizes to 28x28, and outputs predicted class and confidence.
- **Web App (`app/app.py`)**: Streamlit app providing a user interface to draw/upload digits, inspect predictions, and view model statistics.

## Project Structure
```text
handwritten-digit-recognizer/
├── app/
│   └── app.py               # Streamlit web application
├── src/
│   ├── config.py            # Project paths and hyperparameters
│   ├── preprocessing.py     # Data loading and image scaling
│   ├── model.py              # CNN model architecture
│   ├── train.py              # Model training script
│   ├── evaluate.py           # Evaluation script and graph generation
│   └── predict.py            # Image preprocessing and single prediction
├── models/
│   └── digit_cnn.keras      # Saved trained model file
├── plots/                   # Generated evaluation plots and matrix images
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation
```

## Future Improvements
- Add data augmentation (slight rotation, scaling) to improve recognition on irregular handwriting.
- Extend the app to recognize multi-digit numbers instead of single digits.
- Experiment with adding Batch Normalization to speed up training convergence.

## Author
Priyanshu
