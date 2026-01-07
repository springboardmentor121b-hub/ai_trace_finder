import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from utils import corr2d, extract_enhanced_features

# =====================================================
# BASE PATHS (FIXED)
# =====================================================
BASE_DIR = r"D:\TracerFinder"
DATA_DIR = os.path.join(BASE_DIR, "Data_Set")
RES_DIR  = os.path.join(BASE_DIR, "results", "hybrid_cnn")

MODEL_PATH   = os.path.join(RES_DIR, "scanner_hybrid_final.h5")
ENCODER_PATH = os.path.join(RES_DIR, "hybrid_label_encoder.pkl")
SCALER_PATH  = os.path.join(RES_DIR, "hybrid_feat_scaler.pkl")

RES_PATH     = os.path.join(RES_DIR, "official_wiki_residuals.pkl")
FP_PATH      = os.path.join(RES_DIR, "scanner_fingerprints.pkl")
ORDER_NPY    = os.path.join(RES_DIR, "fp_keys.npy")

FEATURES_PATH  = os.path.join(RES_DIR, "features.pkl")
ENHANCED_PATH  = os.path.join(RES_DIR, "enhanced_features.pkl")

SEED = 42
np.random.seed(SEED)

# =====================================================
# LOAD MODEL & PREPROCESSORS
# =====================================================
print(f"Loading model from:\n{MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# =====================================================
# LOAD RESIDUALS
# =====================================================
print("Loading residuals...")
with open(RES_PATH, "rb") as f:
    residuals_dict = pickle.load(f)

# =====================================================
# LOAD FEATURES
# =====================================================
print("Loading pre-computed features...")
with open(FEATURES_PATH, "rb") as f:
    prnu_data = pickle.load(f)

with open(ENHANCED_PATH, "rb") as f:
    enh_data = pickle.load(f)

feats_prnu = prnu_data["features"]
feats_enh  = enh_data["features"]

# =====================================================
# REBUILD DATASET
# =====================================================
X_img, X_feat, y = [], [], []
idx = 0

for dataset_name in ["Official", "Wikipedia"]:
    if dataset_name not in residuals_dict:
        continue

    print(f"Processing {dataset_name}...")
    for scanner, dpi_dict in residuals_dict[dataset_name].items():
        for dpi, res_list in dpi_dict.items():
            for res in res_list:
                X_img.append(res[..., np.newaxis])
                X_feat.append(feats_prnu[idx] + feats_enh[idx])
                y.append(scanner)
                idx += 1

X_img = np.array(X_img, dtype=np.float32)
X_feat = np.array(X_feat, dtype=np.float32)
y = np.array(y)

# =====================================================
# ENCODE & SPLIT (SAME AS TRAINING)
# =====================================================
y_int = label_encoder.transform(y)
y_cat = to_categorical(y_int, num_classes=len(label_encoder.classes_))

_, X_img_te, _, X_feat_te, _, y_te = train_test_split(
    X_img,
    X_feat,
    y_cat,
    test_size=0.2,
    random_state=SEED,
    stratify=y_int
)

y_true = np.argmax(y_te, axis=1)
X_feat_te = scaler.transform(X_feat_te)

print(f"Test samples: {len(X_img_te)}")

# =====================================================
# EVALUATION
# =====================================================
print("Running prediction...")
y_pred_prob = model.predict([X_img_te, X_feat_te], batch_size=16)
y_pred = np.argmax(y_pred_prob, axis=1)

acc = accuracy_score(y_true, y_pred)
print(f"\n🎯 Test Accuracy: {acc*100:.2f}%\n")

print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# =====================================================
# CONFUSION MATRIX
# =====================================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Hybrid CNN Confusion Matrix ({acc*100:.2f}%)")

cm_path = os.path.join(RES_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

print(f"✅ Confusion matrix saved to:\n{cm_path}")
