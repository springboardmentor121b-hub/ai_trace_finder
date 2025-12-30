import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..")
sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

DATA_ROOT = "data/official"          # sirf official
RESULTS_DIR = "results/eda_official"
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ds_off = ImageFolder(root=DATA_ROOT, transform=transform)
    train_loader = DataLoader(ds_off, batch_size=BATCH_SIZE, shuffle=True)

    classes = ds_off.classes
    print("OFFICIAL classes:", classes)
    print("Total official images:", len(ds_off))

    # class distribution
    class_counts = {cls: 0 for cls in classes}
    for imgs, labels in train_loader:
        for lab in labels:
            class_counts[classes[int(lab)]] += 1

    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("OFFICIAL - Class distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "official_class_dist.png"), dpi=300)
    plt.close()

    # samples
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
    plt.savefig(os.path.join(RESULTS_DIR, "official_samples.png"), dpi=300)
    plt.close()

    print("Official EDA saved in", RESULTS_DIR)


if __name__ == "__main__":
    main()
