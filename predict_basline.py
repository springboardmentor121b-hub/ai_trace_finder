import numpy as np
import joblib
import pandas as pd
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy
import os

SCALER_PATH = r"D:\project\Tracefinder\models\scaler.pkl"
MODEL_PATH = r"D:\project\Tracefinder\models\random_forest.pkl"

def load_npy_image(npy_path):
    img = np.load(npy_path)
    img = img.astype(np.float32)
    if img.max() > 1:
        img /= 255.0
    return img

def extract_features(img, file_path):
    h, w = img.shape
    aspect_ratio = w / h
    file_size_kb = os.path.getsize(file_path) / 1024

    pixels = img.flatten()
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)

    ent = entropy(np.histogram(pixels, bins=256, range=(0, 1))[0] + 1e-6)

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

def predict_scanner(npy_path):
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    img = load_npy_image(npy_path)
    features = extract_features(img, npy_path)

    df = pd.DataFrame([features])
    X_scaled = scaler.transform(df)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]

    return prediction, probability

if __name__ == "__main__":
    test_file = r"D:\Project\TraceFinder\data\Canon9000-1\300\s4_1_0.npy"
    pred, prob = predict_scanner(test_file)
    print("Predicted Scanner:", pred)
    print("Probabilities:", prob)
