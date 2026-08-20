"""
app.py — Streamlit Web Application for Handwritten Digit Recognition

A clean, professional interface for the digit recognizer.

Features:
- Home page with project overview
- Digit recognition via canvas drawing or image upload
- Probability distribution visualization
- Model information page

Run:  streamlit run app/app.py

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# Add project root to Python path so we can import src modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import MODEL_PATH, NUM_CLASSES
from src.predict import predict_digit, load_model, preprocess_image


# ─── Page Configuration ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS for Professional Styling ─────────────────────────────────────

st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(120deg, #1e88e5, #7c4dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .main-header p {
        color: #666;
        font-size: 1.1rem;
    }

    /* Result card styling */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .result-card .digit {
        font-size: 5rem;
        font-weight: 900;
        line-height: 1;
    }
    .result-card .confidence {
        font-size: 1.5rem;
        opacity: 0.9;
    }

    /* Info box styling */
    .info-box {
        background: #f0f4ff;
        border-left: 4px solid #1e88e5;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }

    /* Feature card */
    .feature-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .feature-card h3 {
        margin-top: 0.5rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar Navigation ─────────────────────────────────────────────────────

st.sidebar.title("🔢 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "✍️ Recognize Digit", "📊 About Model"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Handwritten Digit Recognizer**\n\n"
    "A CNN-based system trained on the MNIST dataset "
    "to recognize handwritten digits 0–9."
)


# ─── Model Loading (cached) ─────────────────────────────────────────────────

@st.cache_resource
def get_model():
    """Load the model once and cache it across reruns."""
    try:
        return load_model()
    except FileNotFoundError:
        return None


# ─── Home Page ───────────────────────────────────────────────────────────────

if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>🔢 Handwritten Digit Recognizer</h1>
        <p>A Convolutional Neural Network trained to recognize handwritten digits with 99%+ accuracy</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Project overview
    st.header("📋 Project Overview")
    st.write(
        "This project implements a complete handwritten digit recognition system "
        "using deep learning. The system can identify digits (0–9) from hand-drawn "
        "or uploaded images with high accuracy."
    )

    # How it works
    st.header("⚙️ How It Works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h2>✍️</h2>
            <h3>Draw / Upload</h3>
            <p>Draw a digit on the canvas or upload an image</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h2>🔄</h2>
            <h3>Preprocess</h3>
            <p>Image is resized to 28×28 and normalized</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h2>🧠</h2>
            <h3>CNN Prediction</h3>
            <p>Convolutional Neural Network analyzes the image</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="feature-card">
            <h2>✅</h2>
            <h3>Result</h3>
            <p>Predicted digit with confidence score</p>
        </div>
        """, unsafe_allow_html=True)

    # Model info
    st.header("🏗️ Model Architecture")
    st.code("""
    Input (28×28×1)
      → Conv2D(32, 3×3, ReLU)  → MaxPool(2×2)    → 14×14×32
      → Conv2D(64, 3×3, ReLU)  → MaxPool(2×2)    → 7×7×64
      → Flatten                                    → 3136
      → Dense(128, ReLU) → Dropout(0.5)
      → Dense(10, Softmax)                         → Prediction
    """, language="text")

    st.markdown("""
    <div class="info-box">
        <strong>📊 Training Dataset:</strong> MNIST — 60,000 training images + 10,000 test images
        of handwritten digits, widely used as a benchmark in computer vision.
    </div>
    """, unsafe_allow_html=True)


# ─── Digit Recognition Page ─────────────────────────────────────────────────

elif page == "✍️ Recognize Digit":
    st.markdown("""
    <div class="main-header">
        <h1>✍️ Recognize a Digit</h1>
        <p>Draw a digit on the canvas or upload an image to get a prediction</p>
    </div>
    """, unsafe_allow_html=True)

    # Check if model is available
    model = get_model()
    if model is None:
        st.error(
            "⚠️ **No trained model found!**\n\n"
            "Please train the model first by running:\n"
            "```\npython -m src.train\n```"
        )
        st.stop()

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["🎨 Draw on Canvas", "📁 Upload Image"],
        horizontal=True
    )

    col_input, col_result = st.columns([1, 1])

    image_data = None

    with col_input:
        if input_method == "🎨 Draw on Canvas":
            st.subheader("Draw a digit below")
            try:
                from streamlit_drawable_canvas import st_canvas

                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0)",
                    stroke_width=20,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    key="canvas",
                )

                if canvas_result.image_data is not None:
                    image_data = canvas_result.image_data

            except ImportError:
                st.warning(
                    "Canvas drawing requires `streamlit-drawable-canvas`.\n\n"
                    "Install it with:\n"
                    "```\npip install streamlit-drawable-canvas\n```\n\n"
                    "In the meantime, use the **Upload Image** option."
                )

        else:  # Upload Image
            st.subheader("Upload a digit image")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["png", "jpg", "jpeg", "bmp"],
                help="Upload a clear image of a single handwritten digit"
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=280)
                image_data = np.array(image)

    # Predict button
    predict_clicked = st.button("🔮 Predict Digit", use_container_width=True,
                                type="primary")

    with col_result:
        if predict_clicked and image_data is not None:
            # Check if canvas has any drawing
            if input_method == "🎨 Draw on Canvas":
                # Check if canvas has any non-black pixels
                if image_data.max() == 0:
                    st.warning("Please draw a digit on the canvas first!")
                    st.stop()

            with st.spinner("Analyzing..."):
                # Run prediction
                result = predict_digit(image_data, model)

                # Show preprocessed image
                processed = preprocess_image(image_data)
                st.subheader("Preprocessed Image (28×28)")
                fig_proc, ax_proc = plt.subplots(figsize=(3, 3))
                ax_proc.imshow(processed.squeeze(), cmap="gray")
                ax_proc.axis("off")
                st.pyplot(fig_proc)
                plt.close(fig_proc)

                # Show prediction result
                st.markdown(f"""
                <div class="result-card">
                    <div class="digit">{result['digit']}</div>
                    <div class="confidence">Confidence: {result['confidence']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                # Probability bar chart
                st.subheader("📊 Probability Distribution")
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ["#667eea" if i != result["digit"] else "#ff6b6b"
                          for i in range(NUM_CLASSES)]
                bars = ax.bar(range(NUM_CLASSES),
                              result["probabilities"] * 100,
                              color=colors, edgecolor="white", linewidth=0.5)

                # Add percentage labels on bars
                for bar, prob in zip(bars, result["probabilities"]):
                    if prob > 0.01:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.5,
                                f"{prob * 100:.1f}%",
                                ha="center", va="bottom", fontsize=9)

                ax.set_xlabel("Digit", fontsize=12)
                ax.set_ylabel("Probability (%)", fontsize=12)
                ax.set_title("Prediction Probabilities", fontsize=14,
                             fontweight="bold")
                ax.set_xticks(range(NUM_CLASSES))
                ax.set_ylim(0, 105)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        elif predict_clicked:
            st.warning("Please draw a digit or upload an image first!")


# ─── About Model Page ───────────────────────────────────────────────────────

elif page == "📊 About Model":
    st.markdown("""
    <div class="main-header">
        <h1>📊 About the Model</h1>
        <p>Technical details about the CNN architecture and training</p>
    </div>
    """, unsafe_allow_html=True)

    st.header("🏗️ CNN Architecture")
    st.markdown("""
    | Layer | Type | Output Shape | Parameters | Purpose |
    |-------|------|-------------|------------|---------|
    | 1 | Conv2D (32 filters, 3×3) | 28×28×32 | 320 | Detect edges and simple patterns |
    | 2 | MaxPooling2D (2×2) | 14×14×32 | 0 | Reduce spatial dimensions |
    | 3 | Conv2D (64 filters, 3×3) | 14×14×64 | 18,496 | Detect complex features |
    | 4 | MaxPooling2D (2×2) | 7×7×64 | 0 | Further spatial reduction |
    | 5 | Flatten | 3,136 | 0 | Convert to 1D vector |
    | 6 | Dense (128 units) | 128 | 401,536 | Learn classifications |
    | 7 | Dropout (50%) | 128 | 0 | Prevent overfitting |
    | 8 | Dense (10 units, Softmax) | 10 | 1,290 | Output probabilities |
    """)

    st.header("📈 Training Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | Optimizer | Adam |
        | Learning Rate | 0.001 |
        | Batch Size | 128 |
        | Max Epochs | 15 |
        | Early Stopping | Patience = 3 |
        """)
    with col2:
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | Loss Function | Categorical Crossentropy |
        | Input Shape | 28 × 28 × 1 |
        | Output Classes | 10 (digits 0–9) |
        | Dropout Rate | 0.5 |
        | Total Parameters | ~421,642 |
        """)

    # Show evaluation plots if they exist
    st.header("📊 Evaluation Results")
    plots_dir = os.path.join(PROJECT_ROOT, "plots")
    plot_files = {
        "Training History": "training_history.png",
        "Confusion Matrix": "confusion_matrix.png",
        "Correct Predictions": "correct_predictions.png",
        "Incorrect Predictions": "incorrect_predictions.png",
        "Class Distribution": "class_distribution.png",
    }

    for title, filename in plot_files.items():
        path = os.path.join(plots_dir, filename)
        if os.path.exists(path):
            st.subheader(title)
            st.image(path, use_container_width=True)

    if not any(os.path.exists(os.path.join(plots_dir, f))
               for f in plot_files.values()):
        st.info(
            "📝 No evaluation plots found yet. "
            "Run training and evaluation first:\n"
            "```\npython -m src.train\npython -m src.evaluate\n```"
        )

    st.header("📚 About MNIST Dataset")
    st.write(
        "The MNIST (Modified National Institute of Standards and Technology) "
        "database is a large collection of handwritten digits commonly used "
        "for training image processing systems. It contains:\n"
        "- **60,000** training images\n"
        "- **10,000** test images\n"
        "- Each image is **28×28 pixels** in grayscale\n"
        "- Digits are centered and size-normalized"
    )


# ─── Footer ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
    <p>Handwritten Digit Recognizer | Built with TensorFlow & Streamlit</p>
</div>
""", unsafe_allow_html=True)
