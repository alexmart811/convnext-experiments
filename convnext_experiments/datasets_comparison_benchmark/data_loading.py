"""Загрузка ImageFolder для классификации сегментации"""

import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets

from .config import CFG
from .datasets import OxfordPetSegmentation, SubsetWithTransform
from .transforms import get_classification_transforms


def _has_class_subfolders(directory: str) -> bool:
    """True, если в каталоге есть хотя бы одна подпапка (класс)."""
    if not os.path.isdir(directory):
        return False
    return any(os.path.isdir(os.path.join(directory, name)) for name in os.listdir(directory))


def load_classification_dataset(
    data_dir: str, image_size: int = 224, dataset_name: str = "unknown"
) -> Tuple[Dataset, Dataset, Dataset, List[str]]:
    """
    Сбор train/val/test для ImageFolder-структуры.

    Поддерживаются: отдельные train/valid/test; train без valid; один корень с классами.
    Возвращает датасеты и список имён классов.
    """
    train_transform = get_classification_transforms(image_size, is_training=True)
    val_transform = get_classification_transforms(image_size, is_training=False)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    if os.path.exists(train_dir):
        if not _has_class_subfolders(train_dir):
            raise FileNotFoundError(f"В {train_dir} нет подпапок классов.")

        val_ok = os.path.exists(val_dir) and _has_class_subfolders(val_dir)
        test_ok = os.path.exists(test_dir) and _has_class_subfolders(test_dir)

        if test_ok:
            test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
            if val_ok:
                train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
                val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
            else:
                full_train = datasets.ImageFolder(train_dir, transform=None)
                n = len(full_train)
                n_train = int(0.8 * n)
                train_subset, val_subset = random_split(
                    full_train,
                    [n_train, n - n_train],
                    generator=torch.Generator().manual_seed(CFG.training.seed),
                )
                train_dataset = SubsetWithTransform(train_subset, train_transform)
                val_dataset = SubsetWithTransform(val_subset, val_transform)
            class_names = (
                train_dataset.classes if hasattr(train_dataset, "classes") else full_train.classes
            )
        else:
            if os.path.exists(test_dir) and not test_ok:
                print(
                    f"Dataset {dataset_name}: test/ без подпапок классов — "
                    "train/ делится на train/val/test (70/15/15)."
                )
            full_train = datasets.ImageFolder(train_dir, transform=None)
            n = len(full_train)
            n_train = int(0.70 * n)
            n_val = int(0.15 * n)
            n_test = n - n_train - n_val
            if n_test < 1:
                n_test = 1
                n_val = max(1, n_val - 1)
                n_train = n - n_val - n_test

            train_subset, val_subset, test_subset = random_split(
                full_train,
                [n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(CFG.training.seed),
            )
            train_dataset = SubsetWithTransform(train_subset, train_transform)
            val_dataset = SubsetWithTransform(val_subset, val_transform)
            test_dataset = SubsetWithTransform(test_subset, val_transform)
            class_names = full_train.classes
    else:
        full_dataset = datasets.ImageFolder(data_dir, transform=None)
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(CFG.training.seed),
        )
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform
        test_dataset.dataset.transform = val_transform
        class_names = full_dataset.classes

    print(
        f"Dataset: {dataset_name} | Train: {len(train_dataset)}, Val: {len(val_dataset)}, "
        f"Test: {len(test_dataset)} | Classes: {len(class_names)}"
    )
    return train_dataset, val_dataset, test_dataset, class_names


def create_segmentation_dataloaders(
    root_dir: str, image_size: int = 224, batch_size: int = 16
) -> Tuple[DataLoader, DataLoader, OxfordPetSegmentation]:
    """Train/val DataLoader для сегментации и датасет train"""
    train_ds = OxfordPetSegmentation(root_dir, split="train", image_size=image_size)
    val_ds = OxfordPetSegmentation(root_dir, split="val", image_size=image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=CFG.hardware.num_workers,
        pin_memory=CFG.hardware.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=CFG.hardware.num_workers,
        pin_memory=CFG.hardware.pin_memory,
    )

    print(f"Segmentation: Train={len(train_ds)}, Val={len(val_ds)}")
    return train_loader, val_loader, train_ds
