import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from configs.config import RAW_OFFICIAL_DIR, RESULTS_DIR
    DATA_ROOT = str(RAW_OFFICIAL_DIR)
    EDA_RESULTS_DIR = str(RESULTS_DIR / "eda_official")
except ImportError:
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "Official")
    EDA_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "eda_official")

BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def main():
    if not os.path.exists(DATA_ROOT) or len(os.listdir(DATA_ROOT)) == 0:
        print(f"Data root directory '{DATA_ROOT}' is empty or does not exist. Please place official dataset folders in 'data/Official/'.")
        return

    os.makedirs(EDA_RESULTS_DIR, exist_ok=True)

    try:
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
        plt.savefig(os.path.join(EDA_RESULTS_DIR, "official_class_dist.png"), dpi=300)
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
        plt.savefig(os.path.join(EDA_RESULTS_DIR, "official_samples.png"), dpi=300)
        plt.close()

        print("Official EDA saved in", EDA_RESULTS_DIR)
    except Exception as e:
        print(f"EDA skipped or failed: {e}")

if __name__ == "__main__":
    main()
