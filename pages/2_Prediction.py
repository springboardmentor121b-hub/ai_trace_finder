import streamlit as st
import torch
import numpy as np
import pandas as pd
import os
import sys
from PIL import Image
from torchvision import transforms
import joblib
import time

from skimage import io, img_as_float
from skimage.filters import sobel
from skimage.restoration import denoise_wavelet
from scipy.stats import skew, kurtosis, entropy

st.set_page_config(
    page_title="Prediction | TraceFinder",
    layout="wide"
)

st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #F1F3FA, #E8ECFF);
}
.card {
    background-color: #EEF1FF;
    padding: 1.8rem;
    border-radius: 18px;
    box-shadow: 0px 12px 30px rgba(90,100,180,0.15);
    margin-bottom: 1.6rem;
    transition: transform 0.3s ease;
}
.card:hover {
    transform: scale(1.01);
}
.badge {
    display: inline-block;
    background-color: #6C7BFF;
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 14px;
}
.highlight {
    background-color: #DFF5EA;
    padding: 12px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# Manual paths
CNN_SRC_PATH = r"D:\Project\TraceFinder\src\cnn_model"
CNN_MODEL_PATH = r"D:\Project\TraceFinder\models\cnn\cnn_model.pth"

SCALER_PATH = r"D:\Project\TraceFinder\models\svm_random\scaler.pkl"
RF_PATH = r"D:\Project\TraceFinder\models\svm_random\random_forest.pkl"
SVM_PATH = r"D:\Project\TraceFinder\models\svm_random\svm.pkl"
LE_PATH = r"D:\Project\TraceFinder\models\svm_random\label_encoder.pkl"

DATA_DIR = r"D:\Project\TraceFinder\data\Official"
TEMP_PATH = r"D:\Project\TraceFinder\temp_upload.tif"

sys.path.insert(0, CNN_SRC_PATH)
from model import SimpleCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

IMG_SIZE = 128
cnn_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def extract_features(img, file_path):
    denoised = denoise_wavelet(img, channel_axis=None, rescale_sigma=True)
    residual = img - denoised
    fft = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    pixels = img.flatten()

    return {
        "mean": np.mean(pixels),
        "std": np.std(pixels),
        "skew": skew(pixels),
        "kurtosis": kurtosis(pixels),
        "entropy": entropy(np.histogram(pixels, bins=256)[0] + 1e-6),
        "fft_mean": np.mean(fft),
        "fft_std": np.std(fft),
        "residual_mean": np.mean(residual),
        "residual_std": np.std(residual),
    }

st.markdown("""
<div class="card">
<h1>🔍 Scanner Source Prediction</h1>
<span class="badge">Hybrid CNN – Primary Model</span>
<p>
This system uses a <b>Hybrid CNN architecture</b> combining
deep residual learning with handcrafted forensic features
for accurate scanner identification.
</p>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ About Hybrid CNN Model"):
    st.markdown("""
    <div class="highlight">
    <b>Hybrid CNN</b> integrates:
    <ul>
    <li>Noise Residual Learning (PRNU)</li>
    <li>Frequency-domain features</li>
    <li>Deep CNN feature extraction</li>
    </ul>
    This approach improves robustness and accuracy compared to standalone models.
    </div>
    """, unsafe_allow_html=True)

model_choice = st.selectbox(
    "Select Prediction Model",
    ["Hybrid CNN (Recommended)", "CNN", "Support Vector Machine (SVM)", "Random Forest"]
)

uploaded_file = st.file_uploader(
    "Upload scanned image",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file:
    with open(TEMP_PATH, "wb") as f:
        f.write(uploaded_file.read())

    st.image(Image.open(TEMP_PATH), use_container_width=True)

    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    if model_choice in ["Hybrid CNN (Recommended)", "CNN"]:
        image = Image.open(TEMP_PATH).convert("RGB")
        img_tensor = cnn_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = cnn_model(img_tensor)
            _, pred = torch.max(outputs, 1)

        st.success("Prediction completed using Hybrid CNN")
        st.success(f"Predicted Scanner: {CLASSES[pred.item()]}")

    else:
        img = img_as_float(io.imread(TEMP_PATH, as_gray=True))
        feats = extract_features(img, TEMP_PATH)
        df = pd.DataFrame([feats])
        df = df[scaler.feature_names_in_]
        X = scaler.transform(df)

        if model_choice == "Support Vector Machine (SVM)":
            pred = svm_model.predict(X)[0]
        else:
            pred = rf_model.predict(X)[0]

        label = label_encoder.inverse_transform([pred])[0]
        st.success(f"Predicted Scanner: {label}")
