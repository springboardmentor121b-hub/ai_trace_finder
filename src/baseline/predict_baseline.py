import cv2
import os
import numpy as np
import joblib
from skimage.restoration import denoise_wavelet
from skimage.filters import sobel
from scipy.stats import skew, kurtosis, entropy
import pandas as pd

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "baseline")

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
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    le_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
    
    if model_choice == "rf":
        model_path = os.path.join(MODEL_DIR, "random_forest.joblib")
    else:
        model_path = os.path.join(MODEL_DIR, "svm.joblib")

    print(f"Loading resources from {MODEL_DIR}...")
    try:
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)
        model = joblib.load(model_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None

    
    img = load_and_preprocess(img_path)
    features = compute_metadata_features(img, img_path)

    # Ensure feature order matches training
    # Note: This assumes the dictionary order is preserved and matches training columns. 
    # Ideally, we should enforce column order if we knew it, but for now we rely on the dict keys.
    # The training script dropped ["file_name", "main_class", "resolution", "class_label"]
    # The compute_metadata_features returns exactly the features used in training.
    
    df = pd.DataFrame([features])
    X_scaled = scaler.transform(df)

    
    pred_idx = model.predict(X_scaled)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_scaled)[0]
        # Create a dict of class -> probability
        prob_dict = {cls: p for cls, p in zip(le.classes_, prob)}
    else:
        prob_dict = None

    return pred_label, prob_dict


if __name__ == "__main__":
    # Example usage
    test_image = os.path.join(PROJECT_ROOT, "Data_Set/Official/EpsonV39-1/300/s8_1.tif")
    
    if os.path.exists(test_image):
        print(f"Testing with image: {test_image}")
        pred, prob = predict_scanner(test_image, model_choice="rf")
        
        if pred is not None:
            print(f"\nPredicted Scanner: {pred}")
            print("Class Probabilities:")
            for cls, p in prob.items():
                print(f"  {cls}: {p:.4f}")
            print("\nPrediction completed successfully!")
    else:
        print(f"Test image not found: {test_image}")