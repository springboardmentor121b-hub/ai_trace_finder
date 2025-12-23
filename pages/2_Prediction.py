import streamlit as st
import torch
import numpy as np
import pandas as pd
import os
import sys
from PIL import Image
from torchvision import transforms
import joblib

from skimage import io, img_as_float
from skimage.filters import sobel
from skimage.restoration import denoise_wavelet
from scipy.stats import skew, kurtosis, entropy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src", "cnn_model"))

from model import SimpleCNN

IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn", "cnn_model.pth")
SVM_DIR = os.path.join(BASE_DIR, "models", "svm_random")

SCALER_PATH = os.path.join(SVM_DIR, "scaler.pkl")
RF_PATH = os.path.join(SVM_DIR, "random_forest.pkl")
SVM_PATH = os.path.join(SVM_DIR, "svm.pkl")
LE_PATH = os.path.join(SVM_DIR, "label_encoder.pkl")

DATA_DIR = os.path.join(BASE_DIR, "data", "Official")

CLASSES = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

@st.cache_resource
def load_cnn():
    model = SimpleCNN(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_baseline():
    scaler = joblib.load(SCALER_PATH)
    rf = joblib.load(RF_PATH)
    svm = joblib.load(SVM_PATH)
    le = joblib.load(LE_PATH)
    return scaler, rf, svm, le

cnn_model = load_cnn()
scaler, rf_model, svm_model, label_encoder = load_baseline()

cnn_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def extract_features(img, file_path):
    h, w = img.shape
    pixels = img.flatten()

    denoised = denoise_wavelet(img, channel_axis=None, rescale_sigma=True)
    residual = img - denoised
    residual_pixels = residual.flatten()

    fft = np.abs(np.fft.fftshift(np.fft.fft2(img)))

    return {
        "width": w,
        "height": h,
        "aspect_ratio": w / h,
        "file_size_kb": os.path.getsize(file_path) / 1024,
        "mean_intensity": np.mean(pixels),
        "std_intensity": np.std(pixels),
        "skewness": skew(pixels),
        "kurtosis": kurtosis(pixels),
        "entropy": entropy(np.histogram(pixels, bins=256)[0] + 1e-6),
        "edge_density": np.mean(sobel(img) > 0.1),
        "fft_mean": np.mean(fft),
        "fft_std": np.std(fft),
        "residual_mean": np.mean(residual_pixels),
        "residual_std": np.std(residual_pixels),
    }

st.title("Image Prediction")

model_choice = st.selectbox(
    "Select Prediction Model",
    ["CNN", "Support Vector Machine (SVM)", "Random Forest"]
)

uploaded_file = st.file_uploader(
    "Upload image for prediction",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file is not None:
    temp_path = os.path.join(BASE_DIR, "temp_upload.tif")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.subheader("Uploaded Image")

    if model_choice == "CNN":
        image = Image.open(temp_path).convert("RGB")
        st.image(image, use_container_width=True)

        img_tensor = cnn_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = cnn_model(img_tensor)
            _, pred = torch.max(outputs, 1)

        st.success("Model Used: CNN")
        st.success(f"Predicted Class: {CLASSES[pred.item()]}")

    else:
        img = io.imread(temp_path, as_gray=True)
        img = img_as_float(img)
        st.image(img, use_container_width=True)

        features = extract_features(img, temp_path)
        df = pd.DataFrame([features])
        df = df[scaler.feature_names_in_]
        X = scaler.transform(df)

        if model_choice == "Support Vector Machine (SVM)":
            pred = svm_model.predict(X)[0]
            model_name = "SVM"
        else:
            pred = rf_model.predict(X)[0]
            model_name = "Random Forest"

        label = label_encoder.inverse_transform([pred])[0]

        st.success(f"Model Used: {model_name}")
        st.success(f"Predicted Class: {label}")
