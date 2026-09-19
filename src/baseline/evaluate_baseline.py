import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys

# Ensure configs module can be imported
sys.path.append(str(Path(__file__).resolve().parents[2]))
try:
    from configs.config import METADATA_CSV, SCALER_PATH, RF_MODEL_PATH, SVM_MODEL_PATH, RESULTS_DIR
    CSV_PATH = str(METADATA_CSV)
    DEFAULT_RESULTS_DIR = str(RESULTS_DIR)
except ImportError:
    CSV_PATH = "data/metadata_features.csv"
    SCALER_PATH = Path("models/scaler.pkl")
    RF_MODEL_PATH = Path("models/random_forest.pkl")
    SVM_MODEL_PATH = Path("models/svm.pkl")
    DEFAULT_RESULTS_DIR = "results"

def evaluate_model(model_path, name, save_dir=None):
    if save_dir is None:
        save_dir = DEFAULT_RESULTS_DIR

    # Load dataset
    df = pd.read_csv(CSV_PATH)
    feature_cols = [c for c in df.columns if c not in ["file_name", "main_class", "resolution", "class_label"]]
    X = df[feature_cols]
    y = df["class_label"]

    # Load scaler + model
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(model_path)

    # Transform features
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    # Print report
    print(f"\n=== {name} Evaluation ===")
    print(classification_report(y, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=model.classes_,
                yticklabels=model.classes_,
                cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Ensure results directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name.replace(' ', '_')}_confusion_matrix.png")

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f" Confusion matrix saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    evaluate_model(RF_MODEL_PATH, "Random Forest")
    evaluate_model(SVM_MODEL_PATH, "SVM")
