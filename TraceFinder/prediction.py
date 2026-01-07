import numpy as np
import joblib
import cv2
from io import BytesIO
from skimage import io, img_as_float
from skimage.restoration import denoise_wavelet
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy
import pickle
import os

# ---------------- PATHS ---------------- #
BASE = r"D:\TracerFinder"

RF_MODEL_PATH = BASE + r"\models\random_forest.pkl"
SVM_MODEL_PATH = BASE + r"\models\svm.pkl"
SCALER_PATH = BASE + r"\models\scaler.pkl"

CNN_MODEL_PATH = BASE + r"\results\hybrid_cnn\scanner_hybrid_final.h5"
CNN_ENCODER_PATH = BASE + r"\results\hybrid_cnn\hybrid_label_encoder.pkl"
CNN_SCALER_PATH = BASE + r"\results\hybrid_cnn\hybrid_feat_scaler.pkl"


# ---------------- PREPROCESSING ---------------- #
def load_and_preprocess_image(file_bytes, size=(256, 256)):
    img = io.imread(BytesIO(file_bytes), as_gray=True)
    img = img_as_float(img)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def extract_features(img, file_bytes):
    h, w = img.shape
    pixels = img.flatten()

    features = [
        w, h, w / h,
        len(file_bytes) / 1024,
        np.mean(pixels),
        np.std(pixels),
        skew(pixels),
        kurtosis(pixels),
        entropy(np.histogram(pixels, bins=256)[0] + 1e-6),
        np.mean(sobel(img) > 0.1)
    ]

    return np.array(features).reshape(1, -1)


# ---------------- MAIN PREDICTION ---------------- #
def predict_image(file_bytes, model_choice):

    img = load_and_preprocess_image(file_bytes)

    # ---------- RANDOM FOREST ----------
    if model_choice == "Random Forest":
        model = joblib.load(RF_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        feats = extract_features(img, file_bytes)
        feats = scaler.transform(feats)

        pred = model.predict(feats)[0]
        probs = model.predict_proba(feats)[0]

        return pred, probs

    # ---------- SVM ----------
    if model_choice == "SVM":
        model = joblib.load(SVM_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        feats = extract_features(img, file_bytes)
        feats = scaler.transform(feats)

        pred = model.predict(feats)[0]

        return pred, np.array([1.0])  # SVM no prob by default

    # ---------- CNN (LAZY LOAD TF) ----------
    if model_choice == "CNN":
        import tensorflow as tf  # ✅ IMPORT HERE ONLY

        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)

        with open(CNN_ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)

        with open(CNN_SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        img = cv2.resize(img, (256, 256))
        img = np.expand_dims(img, axis=(0, -1))

        dummy_feat = np.zeros((1, scaler.mean_.shape[0]))
        dummy_feat = scaler.transform(dummy_feat)

        probs = cnn_model.predict([img, dummy_feat])[0]
        idx = np.argmax(probs)

        return encoder.classes_[idx], probs
