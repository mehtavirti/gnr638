import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_transforms(img_size=224, augment=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize,
    ]) if augment else transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, val_transform


def get_dataloaders(data_dir, batch_size=64, num_workers=4,
                    seed=42, img_size=224, augment=True,
                    fraction=1.0):
    """
    Main dataloader for all scenarios.
    fraction: use 1.0 / 0.2 / 0.05 for few-shot (Scenario 3)
    """
    train_transform, val_transform = get_transforms(img_size, augment)

    # Full dataset for indexing
    full_dataset = datasets.ImageFolder(root=data_dir)
    total = len(full_dataset)

    np.random.seed(seed)
    indices = list(range(total))
    np.random.shuffle(indices)

    # 80/20 train-val split
    split = int(0.8 * total)
    train_idx = indices[:split]
    val_idx   = indices[split:]

    # Apply fraction for few-shot
    if fraction < 1.0:
        n = max(1, int(len(train_idx) * fraction))
        np.random.seed(seed)
        train_idx = np.random.choice(train_idx, size=n, replace=False).tolist()

    # Train dataset with augmentation
    train_full = datasets.ImageFolder(root=data_dir, transform=train_transform)
    train_dataset = Subset(train_full, train_idx)

    # Val dataset without augmentation
    val_full = datasets.ImageFolder(root=data_dir, transform=val_transform)
    val_dataset = Subset(val_full, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    print(f"[Dataset] Train: {len(train_idx)} | Val: {len(val_idx)} "
          f"| Fraction used: {fraction}")
    return train_loader, val_loader, full_dataset.classes


def get_fixed_pca_subset(data_dir, n_classes=30, n_per_class=30,
                          seed=42, img_size=224, batch_size=64):
    """
    Fixed subset of exactly 30 classes x 30 samples = 900 samples.
    MUST be the same across all models and all layers (Scenario 5).
    """
    _, val_transform = get_transforms(img_size, augment=False)
    dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)

    np.random.seed(seed)
    class_to_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices.setdefault(label, []).append(idx)

    selected_indices = []
    for cls in range(n_classes):
        cls_indices = class_to_indices[cls]
        chosen = np.random.choice(
            cls_indices,
            size=min(n_per_class, len(cls_indices)),
            replace=False
        )
        selected_indices.extend(chosen.tolist())

    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=batch_size,
                        shuffle=False, num_workers=2)
    print(f"[PCA Subset] Total samples: {len(selected_indices)} "
          f"({n_classes} classes x {n_per_class} samples)")
    return loader, dataset.classes