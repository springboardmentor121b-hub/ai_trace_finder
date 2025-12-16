import os
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from dataset import get_dataloaders
from model import SimpleCNN


def evaluate_model(model_path="D:\Project\TraceFinder\models\cnn\cnn_model.pth",
                   batch_size=32,
                   data_path="D:\Project\TraceFinder\data"):

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # results
    results_dir = r"D:\Project\TraceFinder\results"
    os.makedirs(results_dir, exist_ok=True)

    # data
    _, _, test_loader, classes = get_dataloaders(
        data_root=data_path,
        batch_size=batch_size
    )

    # model
    model = SimpleCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # architecture
    with open(os.path.join(results_dir, "architecture.txt"), "w") as f:
        f.write(str(model))

    # evaluation
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # report
    report = classification_report(all_labels, all_preds, target_names=classes)
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # matrix
    cm = confusion_matrix(all_labels, all_preds)
    with open(os.path.join(results_dir, "confusion_matrix.txt"), "w") as f:
        f.write(str(cm))

    # plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=classes,
                yticklabels=classes,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="D:\Project\TraceFinder\models\cnn\cnn_model.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_path", type=str, default="D:\Project\TraceFinder\data")

    args = parser.parse_args()

    # run
    evaluate_model(
        model_path=args.model_path,
        batch_size=args.batch_size,
        data_path=args.data_path
    )
