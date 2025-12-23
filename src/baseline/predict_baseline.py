
import cv2
import os
import numpy as np
import joblib
from skimage.restoration import denoise_wavelet
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy
import pandas as pd

from PIL import Image

# def load_and_preprocess(img_path, size=(512, 512)):
#     try:
#         # Use PIL – safe for TIFF files
#         img = Image.open(img_path)
#         img = img.convert("L")  # convert to grayscale
#         img = np.array(img, dtype=np.float32) / 255.0
#         img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
#         return img
#     except Exception as e:
#         raise ValueError(f"Could not load image: {img_path} -> {str(e)}")


def load_and_preprocess(img_path, size=(512, 512)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
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
    
    scaler = joblib.load("models_official/scaler.pkl")
    if model_choice == "rf":
        model = joblib.load("models_official/random_forest.pkl")
    else:
        model = joblib.load("models_official/svm.pkl")

    
    img = load_and_preprocess(img_path)
    features = compute_metadata_features(img, img_path)

    
    df = pd.DataFrame([features])
    X_scaled = scaler.transform(df)

    
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]

    return pred, prob


if __name__ == "__main__":
    test_image = "../../dataSet/Official/EpsonV39-1/300/s8_1.tif"
#     test_image = os.path.normpath(
#     os.path.join(os.path.dirname(__file__), '..', 'dataSet', 'Official','EpsonV39-1','150','s8_1.tif')
# )
#     test_image = r"F:\springboard\dataSet\Official\EpsonV39-1\300\s8_1.tif"
    pred, prob = predict_scanner(test_image, model_choice="rf")
    print("Predicted Scanner:", pred)
    print("Class Probabilities:", prob)
    print(" Prediction completed successfully!")