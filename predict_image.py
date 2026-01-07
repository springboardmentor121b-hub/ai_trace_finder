import os
import cv2
import numpy as np
import joblib
from scipy.stats import skew, kurtosis
from skimage.filters import sobel

MODEL_PATH = "models/random_forest.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# -------- Feature extractor --------
def extract_features(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"File not found: {img_path}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not readable (check extension & path)")

    edges = sobel(img)

    features = [
        img.shape[1],                 # width
        img.shape[0],                 # height
        img.shape[1] / img.shape[0],  # aspect_ratio
        os.path.getsize(img_path) / 1024,  # file_size_kb
        np.mean(img),
        np.std(img),
        skew(img.flatten()),
        kurtosis(img.flatten()),
        np.mean(edges),
        np.mean(edges > 0)
    ]
    return np.array(features).reshape(1, -1)

# -------- Load models --------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

# -------- Predict --------
img_path = input("Enter image path: ").strip('"')

X = extract_features(img_path)
X_scaled = scaler.transform(X)
pred = model.predict(X_scaled)

label = encoder.inverse_transform(pred)[0]
print("Predicted Source:", label)