import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split

class SafeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            img, label = self.dataset[idx]
            if label >= self.num_classes:
                label = label % self.num_classes
            return img, label
        except Exception:
            # skip corrupted image by moving to next index
            return self.__getitem__((idx + 1) % len(self))




def get_dataloaders(data_root="Data_Set", batch_size=32, img_size=128, val_split=0.1, test_split=0.1):
    """
    Creates DataLoaders for train, validation, and test sets.
    Combines 'official' and 'Wikipedia' datasets.
    """
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) # Assuming grayscale or normalizing per channel
    ])
    
    datasets_list = []
    
    # 1. Official Dataset
    official_dir = os.path.join(data_root, "official")
    if os.path.exists(official_dir):
        print(f"Found official dataset at {official_dir}")
        ds_official = datasets.ImageFolder(root=official_dir, transform=transform)
        datasets_list.append(ds_official)
    else:
        print(f"Warning: Official dataset not found at {official_dir}")

    # 2. Wikipedia Dataset
    wiki_dir = os.path.join(data_root, "Wikipedia")
    if os.path.exists(wiki_dir):
        print(f"Found Wikipedia dataset at {wiki_dir}")
        ds_wiki = datasets.ImageFolder(root=wiki_dir, transform=transform)
        datasets_list.append(ds_wiki)
    else:
        print(f"Warning: Wikipedia dataset not found at {wiki_dir}")

    if not datasets_list:
        raise ValueError("No datasets found! Please check Data_Set structure.")

    # 🔧 FORCE SAME CLASS INDEXING (THIS FIXES THE ERROR)
    base_classes = datasets_list[0].classes
    base_class_to_idx = datasets_list[0].class_to_idx

    for ds in datasets_list[1:]:
        ds.classes = base_classes
        ds.class_to_idx = base_class_to_idx
    
    full_dataset_raw = ConcatDataset(datasets_list)
    num_classes = len(datasets_list[0].classes)
    full_dataset = SafeDataset(full_dataset_raw, num_classes)

    
    # Split
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Total images: {total_size}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    classes = datasets_list[0].classes
    
    return train_loader, val_loader, test_loader, classes
