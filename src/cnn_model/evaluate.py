import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

from .dataset import get_dataloaders
from .model import SimpleCNN


def evaluate_model(
    model_path=r"D:\TracerFinder\models\cnn\cnn_model.pth",
    batch_size=32,
    data_path=r"D:\TracerFinder\cnn_data",
    save_cm=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    print("Loading test data...")
    _, _, test_loader, classes = get_dataloaders(
        data_root=data_path,
        batch_size=batch_size
    )

    num_classes = len(classes)

    # Load model
    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {model_path}")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n🎯 Test Accuracy: {acc * 100:.2f}%\n")

    print("📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    if save_cm:
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("CNN Confusion Matrix")

        save_dir = r"D:\TracerFinder\results"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "CNN_confusion_matrix.png")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Confusion matrix saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CNN Model")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    evaluate_model(batch_size=args.batch_size)
