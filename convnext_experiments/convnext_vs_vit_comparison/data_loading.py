"""Загрузка ImageFolder-датасетов для классификации."""

import os
from typing import List, Tuple

import kagglehub
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets

from .config import CFG
from .transforms import get_train_transforms, get_val_transforms

KAGGLE_DATASETS = {
    "Flowers_Dataset": "imsparsh/flowers-dataset",
    "ImageNet-Mini/images": "deeptrial/miniimagenet",
}


def ensure_dataset(data_path: str, rel_path: str) -> str:
    """
    Если data_path существует — возвращает его.
    Иначе пытается скачать через kagglehub и возвращает путь к скачанному.
    """
    if os.path.exists(data_path):
        return data_path

    kaggle_slug = KAGGLE_DATASETS.get(rel_path)
    if kaggle_slug is None:
        return data_path

    print(f"  Загрузка датасета {kaggle_slug} через kagglehub...")
    downloaded = kagglehub.dataset_download(kaggle_slug)

    candidate = os.path.join(downloaded, rel_path)
    if os.path.exists(candidate):
        return candidate

    for part in rel_path.split("/"):
        test = os.path.join(downloaded, part)
        if os.path.exists(test) and os.path.isdir(test):
            return test

    return downloaded


class SubsetWithTransform(Dataset):
    """Обёртка над Subset: применяет transform к изображению."""

    def __init__(self, subset: Dataset, transform=None) -> None:
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx: int):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return len(self.subset)


def _has_class_subfolders(directory: str) -> bool:
    if not os.path.isdir(directory):
        return False
    return any(os.path.isdir(os.path.join(directory, name)) for name in os.listdir(directory))


def load_dataset(
    data_dir: str, image_size: int = 224, dataset_name: str = "unknown"
) -> Tuple[Dataset, Dataset, Dataset, List[str]]:
    """
    Загрузка train/val/test для ImageFolder-структуры.

    Поддерживает варианты: train/valid/test; train без valid; единый корень с классами.
    Возвращает (train_dataset, val_dataset, test_dataset, class_names).
    """
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    if os.path.exists(train_dir) and _has_class_subfolders(train_dir):
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
                train_sub, val_sub = random_split(
                    full_train,
                    [n_train, n - n_train],
                    generator=torch.Generator().manual_seed(CFG.training.seed),
                )
                train_dataset = SubsetWithTransform(train_sub, train_transform)
                val_dataset = SubsetWithTransform(val_sub, val_transform)
            class_names = (
                train_dataset.classes
                if hasattr(train_dataset, "classes")
                else full_train.classes
            )
        else:
            full_train = datasets.ImageFolder(train_dir, transform=None)
            n = len(full_train)
            n_train = int(0.70 * n)
            n_val = int(0.15 * n)
            n_test = n - n_train - n_val
            if n_test < 1:
                n_test = 1
                n_val = max(1, n_val - 1)
                n_train = n - n_val - n_test

            train_sub, val_sub, test_sub = random_split(
                full_train,
                [n_train, n_val, n_test],
                generator=torch.Generator().manual_seed(CFG.training.seed),
            )
            train_dataset = SubsetWithTransform(train_sub, train_transform)
            val_dataset = SubsetWithTransform(val_sub, val_transform)
            test_dataset = SubsetWithTransform(test_sub, val_transform)
            class_names = full_train.classes
    else:
        full_dataset = datasets.ImageFolder(data_dir, transform=None)
        n = len(full_dataset)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        n_test = n - n_train - n_val

        train_sub, val_sub, test_sub = random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(CFG.training.seed),
        )
        train_dataset = SubsetWithTransform(train_sub, train_transform)
        val_dataset = SubsetWithTransform(val_sub, val_transform)
        test_dataset = SubsetWithTransform(test_sub, val_transform)
        class_names = full_dataset.classes

    print(
        f"  Dataset: {dataset_name} | "
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)} | "
        f"Classes: {len(class_names)}"
    )
    return train_dataset, val_dataset, test_dataset, class_names
