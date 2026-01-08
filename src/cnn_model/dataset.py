import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split


def get_dataloaders(
    data_root="../../datasets",
    batch_size=32,
    img_size=128,
    val_split=0.1,
    test_split=0.1
):
    print("get_dataloaders() called")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    datasets_list = []

    official_dir = os.path.join(data_root, "official")
    if os.path.exists(official_dir):
        print(f"Found official dataset: {official_dir}")
        datasets_list.append(
            datasets.ImageFolder(official_dir, transform=transform)
        )
    else:
        print(f"Official dataset NOT found: {official_dir}")

    wiki_dir = os.path.join(data_root, "wikipedia")
    if os.path.exists(wiki_dir):
        print(f"Found wikipedia dataset: {wiki_dir}")
        datasets_list.append(
            datasets.ImageFolder(wiki_dir, transform=transform)
        )
    else:
        print(f"Wikipedia dataset NOT found: {wiki_dir}")

    if not datasets_list:
        raise RuntimeError("No datasets found!")

    full_dataset = ConcatDataset(datasets_list)

    total = len(full_dataset)
    test_size = int(total * test_split)
    val_size = int(total * val_split)
    train_size = total - test_size - val_size

    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    classes = datasets_list[0].classes

    return train_loader, val_loader, test_loader, classes


print("dataset.py loaded successfully")
