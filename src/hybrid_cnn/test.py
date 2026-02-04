import os
import glob
import pickle
import numpy as np
import tensorflow as tf
import csv

from utils import (
    process_batch_gpu,
    batch_corr_gpu,
    extract_enhanced_features
)

# =========================================================
# 🔧 ABSOLUTE PATH CONFIG (IMPORTANT)
# =========================================================
BASE_DIR = r"D:\TracerFinder\Data_Set"
ART_DIR  = r"D:\TracerFinder\results\hybrid_cnn"

CKPT_PATH    = os.path.join(ART_DIR, "scanner_hybrid_final.h5")
ENCODER_PATH = os.path.join(ART_DIR, "hybrid_label_encoder.pkl")
SCALER_PATH  = os.path.join(ART_DIR, "hybrid_feat_scaler.pkl")
FP_PATH      = os.path.join(ART_DIR, "scanner_fingerprints.pkl")
ORDER_NPY    = os.path.join(ART_DIR, "fp_keys.npy")

IMG_SIZE = (256, 256)

# =========================================================
# 🔹 LOAD MODEL & ARTIFACTS
# =========================================================
print(f"Loading model from:\n{CKPT_PATH}")
hyb_model = tf.keras.models.load_model(CKPT_PATH, compile=False)

print("Loading preprocessing artifacts...")

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(FP_PATH, "rb") as f:
    scanner_fps = pickle.load(f)

fp_keys = np.load(ORDER_NPY, allow_pickle=True).tolist()

print("Artifacts loaded successfully ✔")

# =========================================================
# 🔹 BATCH PREDICTION FUNCTION
# =========================================================
def predict_batch(image_paths):
    """
    Predict a batch of images.
    Returns: [(image_path, predicted_label, confidence), ...]
    """

    # 1️⃣ Residual extraction
    residuals = process_batch_gpu(image_paths)
    if not residuals:
        return []

    residuals = np.array(residuals, dtype=np.float32)

    # 2️⃣ PRNU Correlation features
    corr_features = batch_corr_gpu(residuals, scanner_fps, fp_keys)

    # 3️⃣ Enhanced handcrafted features
    enhanced_feats = [
        extract_enhanced_features(res) for res in residuals
    ]
    enhanced_feats = np.array(enhanced_feats, dtype=np.float32)

    # 4️⃣ Combine & scale features
    features = np.hstack([corr_features, enhanced_feats])
    features = scaler.transform(features)

    # 5️⃣ CNN inference
    X_img = np.expand_dims(residuals, -1)
    probs = hyb_model.predict([X_img, features], verbose=0)

    results = []
    for i, prob in enumerate(probs):
        idx = int(np.argmax(prob))
        label = label_encoder.classes_[idx]
        confidence = float(prob[idx] * 100)
        results.append((image_paths[i], label, confidence))

    return results

# =========================================================
# 🔹 FOLDER PREDICTION
# =========================================================
def predict_folder(
    folder_path,
    output_csv="hybrid_folder_results.csv",
    exts=("*.tif", "*.png", "*.jpg", "*.jpeg")
):
    image_files = []
    for ext in exts:
        image_files.extend(
            glob.glob(os.path.join(folder_path, "**", ext), recursive=True)
        )

    print(f"Found {len(image_files)} images in {folder_path}")

    if not image_files:
        print("❌ No images found.")
        return []

    all_results = []
    BATCH_SIZE = 32

    for i in range(0, len(image_files), BATCH_SIZE):
        batch = image_files[i:i + BATCH_SIZE]
        batch_results = predict_batch(batch)

        for img, label, conf in batch_results:
            print(f"{img} → {label} | {conf:.2f}%")
            all_results.append((img, label, conf))

    # Save CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Predicted_Label", "Confidence (%)"])
        writer.writerows(all_results)

    print(f"\n✅ Predictions saved to: {output_csv}")
    return all_results

# =========================================================
# 🔹 MAIN
# =========================================================
if __name__ == "__main__":
    test_folder = os.path.join(BASE_DIR, "Test_Images")

    if not os.path.exists(test_folder):
        print(f"❌ Test folder not found: {test_folder}")
        print("Create it and add images to test.")
    else:
        predict_folder(test_folder)
