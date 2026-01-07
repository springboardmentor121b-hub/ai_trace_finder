import streamlit as st
import numpy as np
import cv2
import os
import joblib
from scipy.stats import skew, kurtosis
from skimage.filters import sobel

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder

model, scaler, label_encoder = load_models()

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image not readable")

    height, width = img.shape
    resolution = width * height
    aspect_ratio = width / height
    file_size_kb = os.path.getsize(img_path) / 1024

    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    skewness = skew(img.flatten())
    kurt = kurtosis(img.flatten())

    entropy = -np.sum(
        np.histogram(img, bins=256, density=True)[0] *
        np.log2(np.histogram(img, bins=256, density=True)[0] + 1e-9)
    )

    edges = sobel(img)
    edge_density = np.mean(edges > 0)

    # ✅ EXACT 11 FEATURES (ORDER MATTERS)
    features = [
        resolution,
        width,
        height,
        aspect_ratio,
        file_size_kb,
        mean_intensity,
        std_intensity,
        skewness,
        kurt,
        entropy,
        edge_density
    ]

    return np.array(features).reshape(1, -1)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI TraceFinder", layout="centered")
st.title("🔍 AI TraceFinder – Scanner Identification")

uploaded_file = st.file_uploader(
    "Upload a scanned image",
    type=["jpg", "png", "tif"]
)

if uploaded_file:
    temp_path = "temp_input_image.png"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Scan", use_container_width=True)

    if st.button("🚀 Identify Scanner"):
        try:
            X = extract_features(temp_path)
            X_scaled = scaler.transform(X)

            pred = model.predict(X_scaled)
            scanner = label_encoder.inverse_transform(pred)[0]

            st.success(f"🖨️ **Predicted Scanner:** {scanner}")

        except Exception as e:
            st.error(str(e))

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
