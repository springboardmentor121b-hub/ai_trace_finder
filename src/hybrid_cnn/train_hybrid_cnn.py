import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"



import pickle
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

from model import build_hybrid_model

BASE_DIR = r"D:\Project\TraceFinder\data"

RES_PATH = r"D:\Project\TraceFinder\models\hybrid_cnn\official_wiki_residuals.pkl"
FEATURES_PATH = r"D:\Project\TraceFinder\models\hybrid_cnn\features.pkl"
ENHANCED_PATH = r"D:\Project\TraceFinder\models\hybrid_cnn\enhanced_features.pkl"

ART_DIR = r"D:\Project\TraceFinder\models\hybrid_cnn"
os.makedirs(ART_DIR, exist_ok=True)



# Reproducibility

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Device Info

print("Using device: CPU")



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



# Build Dataset 

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

print("Image shape   :", X_img.shape)
print("Feature shape :", X_feat.shape)



# Encode Labels

le = LabelEncoder()
y_int = le.fit_transform(y)
num_classes = len(le.classes_)
y_cat = to_categorical(y_int, num_classes)



# Train / Test Split

X_img_tr, X_img_te, X_feat_tr, X_feat_te, y_tr, y_te = train_test_split(
    X_img,
    X_feat,
    y_cat,
    test_size=0.2,
    random_state=SEED,
    stratify=y_int
)



# Normalize Features

scaler = StandardScaler()
X_feat_tr = scaler.fit_transform(X_feat_tr)
X_feat_te = scaler.transform(X_feat_te)

with open(os.path.join(ART_DIR, "hybrid_label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

with open(os.path.join(ART_DIR, "hybrid_feat_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)



# Build Model

model = build_hybrid_model(
    img_shape=(256, 256, 1),
    feat_shape=(X_feat.shape[1],),
    num_classes=num_classes
)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()



# DATA PIPELINE (FREEZE-PROOF)

BATCH_SIZE = 2   # VERY IMPORTANT

train_ds = tf.data.Dataset.from_tensor_slices(
    ((X_img_tr, X_feat_tr), y_tr)
).batch(BATCH_SIZE).prefetch(1)

val_ds = tf.data.Dataset.from_tensor_slices(
    ((X_img_te, X_feat_te), y_te)
).batch(BATCH_SIZE).prefetch(1)



# Train Model

print("\nStarting training...\n")

history = model.fit(
    train_ds,
    epochs=3,          
    validation_data=val_ds
)



# Save Model & History

model.save(os.path.join(ART_DIR, "scanner_hybrid_final.keras"))

with open(os.path.join(ART_DIR, "hybrid_training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)



# Plot Training Curves

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("Accuracy")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("Loss")
plt.legend()
plt.grid()

plot_path = os.path.join(ART_DIR, "training_plot.png")
plt.savefig(plot_path)

print("\n TRAINING COMPLETED SUCCESSFULLY")
print("Training plot saved to:", plot_path)
