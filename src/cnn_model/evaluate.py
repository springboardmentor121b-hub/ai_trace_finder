import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

from dataset import get_dataloaders
from model import SimpleCNN

# =====================================================
# PATH CONFIG (FIXED)
# =====================================================

BASE_DIR = r"C:\Infosys_Internship"

DEFAULT_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "cnn", "cnn_model.pth"
)

DEFAULT_DATA_PATH = os.path.join(
    BASE_DIR, "Data_Set"
)

# =====================================================
# EVALUATION FUNCTION
# =====================================================

def evaluate_model(model_path=DEFAULT_MODEL_PATH,
                   batch_size=32,
                   data_path=DEFAULT_DATA_PATH):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Load Data
    # -----------------------------
    print("Loading test data...")
    _, _, test_loader, classes = get_dataloaders(
        data_root=data_path,
        batch_size=batch_size
    )
    num_classes = len(classes)

    # -----------------------------
    # Load Model
    # -----------------------------
    model = SimpleCNN(num_classes=num_classes).to(device)

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded model from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    model.eval()
    all_preds = []
    all_labels = []

    # -----------------------------
    # Inference
    # -----------------------------
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # -----------------------------
    # Metrics
    # -----------------------------
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)

    print("📉 Confusion Matrix:")
    print(cm)

    # -----------------------------
    # Confusion Matrix Plot
    # -----------------------------
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=classes,
                yticklabels=classes,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    save_path = os.path.join(BASE_DIR, "results", "cnn_confusion_matrix.png")
    plt.savefig(save_path)
    print(f"✅ Confusion matrix saved to {save_path}")
    plt.show()

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CNN Model")
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to saved model (.pth)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Dataset root path"
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        batch_size=args.batch_size,
        data_path=args.data_path
    )
