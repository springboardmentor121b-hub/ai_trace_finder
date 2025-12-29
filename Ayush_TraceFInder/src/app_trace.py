# app_tracefinder.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import cv2
import os
from io import BytesIO
from PIL import Image
from skimage import io as skio, img_as_float
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy

# =============================
# Page config
# =============================
st.set_page_config(
    page_title="TraceFinder — Scanner ID",
    layout="wide"
)

st.title("TraceFinder — Scanner / Camera Source Identifier")
st.markdown(
    "Upload image(s) and identify the **source scanner** using trained models."
)

# =============================
# Feature extraction (Baseline)
# =============================
def load_img_gray(file_bytes, size=(512, 512)):
    img = skio.imread(BytesIO(file_bytes.read()), as_gray=True)
    img = img_as_float(img)
    return cv2.resize(img.astype("float32"), size)

def extract_metadata_features(img, file_size_bytes):
    h, w = img.shape
    pixels = img.flatten()
    edges = sobel(img)

    features = [
        w,                          # width
        h,                          # height
        w / h if h else 0,          # aspect ratio
        file_size_bytes / 1024,     # file size KB
        pixels.mean(),              # mean
        pixels.std(),               # std
        skew(pixels),               # skewness
        kurtosis(pixels),           # kurtosis
        entropy(np.histogram(pixels, bins=256)[0] + 1e-6),
        np.mean(edges > 0.1)        # edge density
    ]
    return np.array([features])

FEATURE_COLUMNS = [
    "width", "height", "aspect_ratio", "file_size_kb",
    "mean_intensity", "std_intensity", "skewness",
    "kurtosis", "entropy", "edge_density"
]

# =============================
# Load Models
# =============================
@st.cache_resource
def load_random_forest():
    base = "models/baseline"
    rf = joblib.load(os.path.join(base, "random_forest.joblib"))
    scaler = joblib.load(os.path.join(base, "scaler.joblib"))
    encoder = joblib.load(os.path.join(base, "label_encoder.joblib"))
    return rf, scaler, encoder

@st.cache_resource
def load_hybrid_cnn():
    from tensorflow.keras.models import load_model
    return load_model("models/hybrid_cnn/scanner_hybrid_final.keras")

# =============================
# Model selector
# =============================
model_choice = st.selectbox(
    "Choose Model",
    ["Baseline – Random Forest", "Hybrid CNN (PRNU-based)"]
)

rf_model = scaler = label_encoder = None
hybrid_model = None

if model_choice == "Baseline – Random Forest":
    rf_model, scaler, label_encoder = load_random_forest()
else:
    hybrid_model = load_hybrid_cnn()

# =============================
# File uploader
# =============================
uploaded_files = st.file_uploader(
    "Upload images",
    type=["tif", "tiff", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# =============================
# Inference
# =============================
if uploaded_files:
    results = []

    for file in uploaded_files:
        img = load_img_gray(BytesIO(file.read()))
        X = extract_metadata_features(img, file.size)

        # -------------------------
        # Random Forest
        # -------------------------
        if model_choice == "Baseline – Random Forest":
            X_scaled = scaler.transform(X)
            probs = rf_model.predict_proba(X_scaled)[0]
            pred_idx = np.argmax(probs)
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            classes = label_encoder.classes_

        # -------------------------
        # Hybrid CNN
        # -------------------------
        else:
            probs = hybrid_model.predict(X, verbose=0)[0]
            pred_idx = np.argmax(probs)
            pred_label = f"Class_{pred_idx}"
            classes = [f"Class_{i}" for i in range(len(probs))]

        row = {
            "File": file.name,
            "Predicted Scanner": pred_label
        }

        top3_idx = np.argsort(probs)[-3:][::-1]

        for i in top3_idx:
            row[f"Top-{len(row)}"] = f"{classes[i]} : {probs[i]:.3f}"


        results.append(row)

    df = pd.DataFrame(results)

    st.markdown("### Results")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download Results CSV",
        df.to_csv(index=False),
        "tracefinder_results.csv",
        "text/csv"
    )

else:
    st.info("Upload image files to start prediction.")
