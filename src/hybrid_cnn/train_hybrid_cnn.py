# src/hybrid_cnn/train_hybrid_cnn.py

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from model import build_hybrid_model

RES_PATH  = "results/hybrid_cnn/official_wiki_residuals.pkl"
FEAT_PATH = "results/hybrid_cnn/features.pkl"
OUT_DIR   = "results/hybrid_cnn"

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------
# Load residuals
# -------------------------------
print("Loading residuals...")
with open(RES_PATH, "rb") as f:
    residuals = pickle.load(f)

# -------------------------------
# Load features
# -------------------------------
print("Loading features...")
with open(FEAT_PATH, "rb") as f:
    data = pickle.load(f)
    X_feat = np.array(data["features"], dtype=np.float32)   # ✅ FIX 1
    y = np.array(data["labels"])

# -------------------------------
# Build image array safely
# -------------------------------
X_img = []

for dataset in ["official", "Wikipedia"]:
    if dataset not in residuals:
        continue

    print(f"Processing {dataset}...")
    for scanner, value in residuals[dataset].items():

        # Case 1: dict (dpi folders)
        if isinstance(value, dict):
            for _, res_list in value.items():
                for res in res_list:
                    X_img.append(np.expand_dims(res, -1))

        # Case 2: list (direct images)
        elif isinstance(value, list):
            for res in value:
                X_img.append(np.expand_dims(res, -1))

X_img = np.array(X_img, dtype=np.float32)   # ✅ FIX 2

print("Images:", X_img.shape)
print("Features:", X_feat.shape)

# -------------------------------
# Encode labels
# -------------------------------
le = LabelEncoder()
y_int = le.fit_transform(y)
y_cat = to_categorical(y_int)

# -------------------------------
# Train / Test split
# -------------------------------
X_img_tr, X_img_te, X_feat_tr, X_feat_te, y_tr, y_te = train_test_split(
    X_img, X_feat, y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y_int
)

# -------------------------------
# Scale features
# -------------------------------
scaler = StandardScaler()
X_feat_tr = scaler.fit_transform(X_feat_tr)
X_feat_te = scaler.transform(X_feat_te)

# Save preprocessors
with open(f"{OUT_DIR}/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open(f"{OUT_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# -------------------------------
# Build & train Hybrid CNN
# -------------------------------
model = build_hybrid_model(
    img_shape=(256, 256, 1),
    feat_shape=(X_feat.shape[1],),
    num_classes=y_cat.shape[1]
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    [X_img_tr, X_feat_tr],
    y_tr,
    validation_data=([X_img_te, X_feat_te], y_te),
    epochs=30,
    batch_size=32
)

model.save(f"{OUT_DIR}/hybrid_model.keras")
print("🎉 Training completed successfully")
