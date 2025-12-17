import numpy as np
import pandas as pd
import joblib
import os

from skimage import io, img_as_float
from skimage.filters import sobel
from skimage.restoration import denoise_wavelet
from scipy.stats import skew, kurtosis, entropy


# ===================== PATHS =====================

MODEL_DIR = r"D:\Project\TraceFinder\models\svm_random"

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")   # change to svm.pkl if needed
LE_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


# ===================== IMAGE LOADING =====================

def load_tif_image(tif_path):
    img = io.imread(tif_path, as_gray=True)
    img = img_as_float(img)
    return img


# ===================== FEATURE EXTRACTION =====================

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


# ===================== PREDICTION =====================

def predict_scanner(tif_path):
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LE_PATH)

    img = load_tif_image(tif_path)
    features = extract_features(img, tif_path)

    df = pd.DataFrame([features])

    # 🔥 ENSURE SAME FEATURE ORDER AS TRAINING
    df = df[scaler.feature_names_in_]

    X_scaled = scaler.transform(df)

    pred_encoded = model.predict(X_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    probabilities = model.predict_proba(X_scaled)[0]

    return pred_label, probabilities


# ===================== MAIN =====================

if __name__ == "__main__":
    test_file = r"D:\Project\TraceFinder\data\Official\Canon120-1\150\s1_1.tif"

    prediction, probs = predict_scanner(test_file)

    print("Predicted Scanner :", prediction)
    print("Class Probabilities:", probs)
