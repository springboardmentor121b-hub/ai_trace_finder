import streamlit as st
import numpy as np
import joblib
import os
import cv2
from io import BytesIO
from pathlib import Path
import sys
from skimage import io, img_as_float
from skimage.restoration import denoise_wavelet
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy

# Ensure configs module can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from configs.config import RF_MODEL_PATH, SCALER_PATH
except ImportError:
    RF_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "random_forest.pkl"
    SCALER_PATH = Path(__file__).resolve().parents[1] / "models" / "scaler.pkl"

st.title("🔍 Predict Scanner from Image")

# -------------------- LOAD MODELS -------------------- #
try:
    rf_model = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_loaded = True
except Exception as e:
    st.error(f"❗ Model files not found or failed to load: {e}")
    model_loaded = False


# -------------------- PREPROCESSING -------------------- #
def load_and_preprocess_image(file_bytes, size=(512, 512)):
    """Read image from bytes and preprocess."""
    img = io.imread(BytesIO(file_bytes), as_gray=True)
    img = img_as_float(img)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def extract_noise_residual(img):
    denoised = denoise_wavelet(img, channel_axis=None, rescale_sigma=True)
    return img - denoised


def compute_metadata_features_for_prediction(img, file_bytes):
    h, w = img.shape
    aspect_ratio = w / h

    file_size_kb = len(file_bytes) / 1024

    pixels = img.flatten()
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)
    ent = entropy(np.histogram(pixels, bins=256, range=(0, 1))[0] + 1e-6)

    edges = sobel(img)
    edge_density = np.mean(edges > 0.1)

    return np.array([
        w, h, aspect_ratio, file_size_kb,
        mean_intensity, std_intensity,
        skewness, kurt, ent, edge_density
    ]).reshape(1, -1)


# -------------------- UI UPLOAD -------------------- #
uploaded_file = st.file_uploader("📤 Upload a scanned image", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file and model_loaded:

    # Save uploaded bytes
    file_bytes = uploaded_file.getvalue()

    # Load & preprocess
    img = load_and_preprocess_image(file_bytes)

    # Display image preview
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Extract features
    features = compute_metadata_features_for_prediction(img, file_bytes)

    # Scale features
    scaled_features = scaler.transform(features)

    # Prediction
    prediction = rf_model.predict(scaled_features)[0]
    probabilities = rf_model.predict_proba(scaled_features)[0]

    st.markdown("---")
    st.success(f"**Predicted Scanner: {prediction}**")
    st.write("### 📊 Class Probabilities")
    st.write(probabilities)
