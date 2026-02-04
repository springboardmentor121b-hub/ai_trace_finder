import os
import pickle
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical

from utils import corr2d, extract_enhanced_features
from model import build_hybrid_model

# ---- Local Paths ----
BASE_DIR = r"D:\TracerFinder\Data_Set"

RES_PATH  = os.path.join(BASE_DIR, "../results/hybrid_cnn/official_wiki_residuals.pkl")
FP_PATH   = os.path.join(BASE_DIR, "../results/hybrid_cnn/scanner_fingerprints.pkl")
ORDER_NPY = os.path.join(BASE_DIR, "../results/hybrid_cnn/fp_keys.npy")
FEATURES_PATH = os.path.join(BASE_DIR, "../results/hybrid_cnn/features.pkl")
ENHANCED_PATH = os.path.join(BASE_DIR, "../results/hybrid_cnn/enhanced_features.pkl")

ART_DIR = r"D:\TracerFinder\results\hybrid_cnn"
os.makedirs(ART_DIR, exist_ok=True)

# ---- Reproducibility ----
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---- Device ----
gpus = tf.config.list_physical_devices('GPU')
device_name = '/GPU:0' if gpus else '/CPU:0'
print("Using device:", device_name)

# ---- Load residuals ----
print("Loading residuals...")
with open(RES_PATH, "rb") as f:
    residuals_dict = pickle.load(f)

# ---- Load precomputed features ----
print("Loading pre-computed features...")
with open(FEATURES_PATH, "rb") as f:
    feats_prnu = pickle.load(f)["features"]

with open(ENHANCED_PATH, "rb") as f:
    feats_enh = pickle.load(f)["features"]

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

X_img = np.array(X_img, dtype=np.float32)
X_feat = np.array(X_feat, dtype=np.float32)
y = np.array(y)

print(f"Dataset Shape: Images {X_img.shape}, Features {X_feat.shape}, Labels {y.shape}")

# ---- Encode labels ----
le = LabelEncoder()
y_int = le.fit_transform(y)
num_classes = len(le.classes_)
y_cat = to_categorical(y_int, num_classes)

# ---- Train-test split ----
X_img_tr, X_img_te, X_feat_tr, X_feat_te, y_tr, y_te = train_test_split(
    X_img, X_feat, y_cat, test_size=0.2, random_state=SEED, stratify=y_int
)

# ---- Normalize features ----
scaler = StandardScaler()
X_feat_tr = scaler.fit_transform(X_feat_tr)
X_feat_te = scaler.transform(X_feat_te)

# ---- Save preprocessors ----
with open(os.path.join(ART_DIR, "hybrid_label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

with open(os.path.join(ART_DIR, "hybrid_feat_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# ---- Build model ----
with tf.device(device_name):
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

    # ---- Data pipeline ----
    BATCH = 32
    train_ds = tf.data.Dataset.from_tensor_slices(
        ((X_img_tr, X_feat_tr), y_tr)
    ).shuffle(len(y_tr)).batch(BATCH).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        ((X_img_te, X_feat_te), y_te)
    ).batch(BATCH).prefetch(tf.data.AUTOTUNE)

    # ---- Callbacks (FIXED) ----
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor="val_accuracy"
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=5, min_lr=1e-6, monitor="val_accuracy"
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ART_DIR, "scanner_hybrid_best.h5"),
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    # ---- Train ----
    history = model.fit(
        train_ds,
        epochs=50,
        validation_data=val_ds,
        callbacks=callbacks
    )

    # ---- Save final model ----
    model.save(os.path.join(ART_DIR, "scanner_hybrid_final.h5"))

    with open(os.path.join(ART_DIR, "hybrid_training_history.pkl"), "wb") as f:
        pickle.dump(history.history, f)

# ---- Plot results ----
def plot_history(hist, save_dir):
    acc = hist.history["accuracy"]
    val_acc = hist.history["val_accuracy"]
    loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "bo-", label="Train Acc")
    plt.plot(epochs, val_acc, "r*-", label="Val Acc")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "bo-", label="Train Loss")
    plt.plot(epochs, val_loss, "r*-", label="Val Loss")
    plt.legend()
    plt.grid()

    path = os.path.join(save_dir, "training_plot.png")
    plt.savefig(path)
    print("Training plots saved to:", path)

plot_history(history, ART_DIR)
print("Training complete!")
