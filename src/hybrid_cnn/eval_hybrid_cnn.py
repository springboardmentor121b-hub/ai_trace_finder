
# CPU-SAFE SETTINGS 

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"



# Imports

import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical



# PATH CONFIGURATION (FIXED)

ART_DIR = r"D:\Project\TraceFinder\models\hybrid_cnn"

MODEL_PATH   = os.path.join(ART_DIR, "scanner_hybrid_final.keras")
ENCODER_PATH = os.path.join(ART_DIR, "hybrid_label_encoder.pkl")
SCALER_PATH  = os.path.join(ART_DIR, "hybrid_feat_scaler.pkl")

RES_PATH      = os.path.join(ART_DIR, "official_wiki_residuals.pkl")
FEATURES_PATH = os.path.join(ART_DIR, "features.pkl")
ENHANCED_PATH = os.path.join(ART_DIR, "enhanced_features.pkl")



# Reproducibility

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)



# Load Trained Model & Tools

print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)



# Load Residuals

print("Loading residuals...")
with open(RES_PATH, "rb") as f:
    residuals_dict = pickle.load(f)



# Load Precomputed Features

print("Loading precomputed features...")
with open(FEATURES_PATH, "rb") as f:
    feats_prnu = pickle.load(f)["features"]

with open(ENHANCED_PATH, "rb") as f:
    feats_enh = pickle.load(f)["features"]



# Rebuild Dataset 

X_img, X_feat, y = [], [], []
idx = 0

for dataset_name in ["Official", "Wikipedia"]:
    if dataset_name not in residuals_dict:
        continue

    print(f"Processing {dataset_name}...")
    for scanner, dpi_dict in residuals_dict[dataset_name].items():
        for dpi, res_list in dpi_dict.items():
            for res in res_list:
                X_img.append(np.expand_dims(res, -1))
                X_feat.append(feats_prnu[idx] + feats_enh[idx])
                y.append(scanner)
                idx += 1

X_img = np.asarray(X_img, dtype=np.float32)
X_feat = np.asarray(X_feat, dtype=np.float32)
y = np.asarray(y)

print("Full dataset size:", X_img.shape[0])



# Encode Labels

y_int = le.transform(y)
num_classes = len(le.classes_)
y_cat = to_categorical(y_int, num_classes)



# Recover SAME test split as training

print("Splitting to recover test set...")
_, X_img_te, _, X_feat_te, _, y_te = train_test_split(
    X_img,
    X_feat,
    y_cat,
    test_size=0.2,
    random_state=SEED,
    stratify=y_int
)

y_int_te = np.argmax(y_te, axis=1)
X_feat_te = scaler.transform(X_feat_te)

print("Test samples:", len(X_img_te))



# Predict (BATCHED – MEMORY SAFE)

print("Running evaluation...")
y_pred_prob = model.predict(
    [X_img_te, X_feat_te],
    batch_size=2,
    verbose=1
)

y_pred = np.argmax(y_pred_prob, axis=1)



# Metrics

test_acc = accuracy_score(y_int_te, y_pred)
print(f"\n Test Accuracy: {test_acc * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_int_te, y_pred, target_names=le.classes_))

# Confusion Matrix

cm = confusion_matrix(y_int_te, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (Accuracy: {test_acc * 100:.2f}%)")

cm_path = os.path.join(ART_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

print(f"\nConfusion matrix saved to:\n{cm_path}")
print("\n EVALUATION COMPLETED SUCCESSFULLY")
