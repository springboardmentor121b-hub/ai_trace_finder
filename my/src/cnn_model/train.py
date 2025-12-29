import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import SimpleCNN


def train_model(
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    data_path: str = "data",
    results_dir: str = "results/cnn",
):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # folders
    os.makedirs("models/cnn", exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join("models", "cnn", "cnn_model.pth")

    # data
    print("Preparing data...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_root=data_path, batch_size=batch_size
    )
    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {classes}")

    # model, loss, optimizer
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0

    # store losses for plotting
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        # -------- TRAIN --------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=float(loss.item()))

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        train_losses.append(avg_train_loss)
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # -------- VALIDATION --------
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_running_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        val_losses.append(avg_val_loss)
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")

    print("\nTraining Complete.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # -------- PLOT LOSS CURVE --------
    epochs_range = range(1, epochs + 1)
    plt.figure()
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    loss_plot_path = os.path.join(results_dir, "loss_curve.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"Loss curve saved to {loss_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN Model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data_path", type=str, default="data", help="Data root path")

    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_path=args.data_path,
    )
