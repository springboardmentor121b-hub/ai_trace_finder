import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_dataloaders(
    data_root=r"D:\TracerFinder\cnn_data",
    batch_size=32,
    img_size=128,
    val_split=0.1,
    test_split=0.1
):
    """
    Creates DataLoaders for train, validation, and test sets
    using a FLAT scanner-wise dataset (ImageFolder compatible).
    """

    # --------------------
    # Transforms
    # --------------------
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # grayscale-safe
    ])

    # --------------------
    # Load dataset
    # --------------------
    if not os.path.exists(data_root):
        raise ValueError(f"Dataset path not found: {data_root}")

    print(f"Loading CNN dataset from: {data_root}")

    full_dataset = datasets.ImageFolder(
        root=data_root,
        transform=transform
    )

    if len(full_dataset) == 0:
        raise ValueError("No images found in cnn_data folders!")

    classes = full_dataset.classes
    print(f"Detected classes ({len(classes)}): {classes}")

    # --------------------
    # Split dataset
    # --------------------
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Total images: {total_size}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # --------------------
    # DataLoaders
    # --------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, classes
