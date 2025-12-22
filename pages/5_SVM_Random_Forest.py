import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

from skimage import io, img_as_float
from skimage.filters import sobel
from skimage.restoration import denoise_wavelet
from scipy.stats import skew, kurtosis, entropy

# -------------------------------------------------
# PATHS (MATCH YOUR PROJECT)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models", "svm_random")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
RF_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
SVM_PATH = os.path.join(MODEL_DIR, "svm.pkl")
LE_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def load_models():
    scaler = joblib.load(SCALER_PATH)
    rf = joblib.load(RF_PATH)
    svm = joblib.load(SVM_PATH)
    le = joblib.load(LE_PATH)
    return scaler, rf, svm, le

scaler, rf_model, svm_model, label_encoder = load_models()

# -------------------------------------------------
# FEATURE EXTRACTION (SAME AS TRAINING)
# -------------------------------------------------
def load_tif_image(path):
    img = io.imread(path, as_gray=True)
    return img_as_float(img)

def extract_noise_residual(img):
    denoised = denoise_wavelet(
        img,
        channel_axis=None,
        rescale_sigma=True
    )
    return img - denoised

def extract_fft_features(img):
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    return np.mean(magnitude), np.std(magnitude)

def extract_features(img, file_path):
    h, w = img.shape
    pixels = img.flatten()

    residual = extract_noise_residual(img)
    residual_pixels = residual.flatten()

    fft_mean, fft_std = extract_fft_features(img)

    features = {
        "width": w,
        "height": h,
        "aspect_ratio": w / h,
        "file_size_kb": os.path.getsize(file_path) / 1024,

        "mean_intensity": np.mean(pixels),
        "std_intensity": np.std(pixels),
        "skewness": skew(pixels),
        "kurtosis": kurtosis(pixels),

        "entropy": entropy(
            np.histogram(pixels, bins=256, range=(0, 1))[0] + 1e-6
        ),

        "edge_density": np.mean(sobel(img) > 0.1),

        "fft_mean": fft_mean,
        "fft_std": fft_std,

        "residual_mean": np.mean(residual_pixels),
        "residual_std": np.std(residual_pixels)
    }

    return features

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("SVM and Random Forest Prediction")

st.write(
    "This page uses classical machine learning models trained on "
    "handcrafted image features such as noise residuals, FFT statistics, "
    "and texture measures."
)

classifier = st.selectbox(
    "Select Model",
    ["Random Forest", "Support Vector Machine"]
)

uploaded_file = st.file_uploader(
    "Upload TIFF / Image File",
    type=["tif", "tiff", "png", "jpg", "jpeg"]
)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if uploaded_file is not None:
    try:
        temp_path = os.path.join(BASE_DIR, "temp_upload.tif")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        img = load_tif_image(temp_path)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        features = extract_features(img, temp_path)
        df = pd.DataFrame([features])

        # 🔥 enforce SAME feature order
        df = df[scaler.feature_names_in_]
        X_scaled = scaler.transform(df)

        if classifier == "Random Forest":
            pred_enc = rf_model.predict(X_scaled)[0]
            probs = rf_model.predict_proba(X_scaled)[0]
            model_name = "Random Forest"
        else:
            pred_enc = svm_model.predict(X_scaled)[0]
            probs = svm_model.predict_proba(X_scaled)[0]
            model_name = "Support Vector Machine"

        pred_label = label_encoder.inverse_transform([pred_enc])[0]

        st.subheader("Prediction Result")
        st.success(f"Model Used: {model_name}")
        st.success(f"Predicted Class: {pred_label}")

        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({
            "Class": label_encoder.classes_,
            "Probability": probs
        })
        st.dataframe(prob_df)

    except Exception as e:
        st.error("Failed to process image. Please upload a valid file.")
