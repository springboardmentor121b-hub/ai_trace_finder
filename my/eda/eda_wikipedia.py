# eda/eda_wikipedia.py
import os
import sys

# project root add
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..")
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms


DATA_ROOT = "data/wikipedia"          # yaha wiki dataset ka folder
RESULTS_DIR = "results/eda_wikipedia"
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def main():
    # folder ensure
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Wikipedia dataset load
    ds_wiki = ImageFolder(root=DATA_ROOT, transform=transform)
    train_loader = DataLoader(ds_wiki, batch_size=BATCH_SIZE, shuffle=True)

    classes = ds_wiki.classes
    print("WIKIPEDIA classes:", classes)
    print("Total wikipedia images:", len(ds_wiki))

    # 2) Class distribution
    class_counts = {cls: 0 for cls in classes}
    for imgs, labels in train_loader:
        for lab in labels:
            class_counts[classes[int(lab)]] += 1

    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("WIKIPEDIA - Class distribution")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "wikipedia_class_dist.png")
    print("Saving class dist to:", out_path)
    plt.savefig(out_path, dpi=300)
    plt.close()

    # 3) Sample images grid
    imgs, labels = next(iter(train_loader))
    n_show = min(16, imgs.size(0))
    rows, cols = 4, 4

    plt.figure(figsize=(8, 8))
    for i in range(n_show):
        plt.subplot(rows, cols, i + 1)
        img = imgs[i].permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.axis("off")
        plt.title(classes[int(labels[i])], fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "wikipedia_samples.png")
    print("Saving samples to:", out_path)
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Wikipedia EDA saved in", RESULTS_DIR)


if __name__ == "__main__":
    main()
