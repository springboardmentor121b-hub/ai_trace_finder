import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from tqdm import tqdm

# ✅ Relative imports (IMPORTANT)
from .dataset import get_dataloaders
from .model import SimpleCNN


def train_model(
    epochs=10,
    batch_size=32,
    learning_rate=0.001,
    data_path=r"D:\TracerFinder\cnn_data"   # ✅ FLAT CNN DATA
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================
    # LOAD DATA
    # =========================
    print("Preparing data...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_root=data_path,
        batch_size=batch_size
    )

    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {classes}")

    # =========================
    # MODEL
    # =========================
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # =========================
    # SAVE PATH (SAFE)
    # =========================
    save_dir = r"D:\TracerFinder\models\cnn"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cnn_model.pth")

    best_val_acc = 0.0

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc="Training", leave=False)

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item())

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # =========================
        # SAVE BEST MODEL
        # =========================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"✅ New best model saved at: {save_path}")

    print("\n🎯 Training Complete.")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN Model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
