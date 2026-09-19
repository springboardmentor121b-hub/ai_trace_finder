import cv2
import os
import numpy as np
import joblib
from pathlib import Path
import sys
from skimage.restoration import denoise_wavelet
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy
import pandas as pd

# Ensure configs module can be imported
sys.path.append(str(Path(__file__).resolve().parents[2]))
try:
    from configs.config import SCALER_PATH, RF_MODEL_PATH, SVM_MODEL_PATH
except ImportError:
    SCALER_PATH = Path("models/scaler.pkl")
    RF_MODEL_PATH = Path("models/random_forest.pkl")
    SVM_MODEL_PATH = Path("models/svm.pkl")

def load_and_preprocess(img_path, size=(512, 512)):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f" Could not load image: {img_path}")
    img = img.astype(np.float32) / 255.0
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def compute_metadata_features(img, file_path):
    h, w = img.shape
    aspect_ratio = w / h
    file_size_kb = os.path.getsize(file_path) / 1024

    pixels = img.flatten()
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)
    ent = entropy(np.histogram(pixels, bins=256, range=(0,1))[0] + 1e-6)

    edges = sobel(img)
    edge_density = np.mean(edges > 0.1)

    return {
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent,
        "edge_density": edge_density
    }

def predict_scanner(img_path, model_choice="rf"):
    scaler = joblib.load(SCALER_PATH)
    if model_choice == "rf":
        model = joblib.load(RF_MODEL_PATH)
    else:
        model = joblib.load(SVM_MODEL_PATH)

    img = load_and_preprocess(img_path)
    features = compute_metadata_features(img, img_path)

    df = pd.DataFrame([features])
    X_scaled = scaler.transform(df)

    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]

    return pred, prob

if __name__ == "__main__":
    test_image = "data/Official/EpsonV39-1/300/s8_1.tif"  
    if os.path.exists(test_image):
        pred, prob = predict_scanner(test_image, model_choice="rf")
        print("Predicted Scanner:", pred)
        print("Class Probabilities:", prob)
        print(" Prediction completed successfully!")
    else:
        print(f"Sample test image {test_image} not found. Ready for predictions upon dataset upload.")