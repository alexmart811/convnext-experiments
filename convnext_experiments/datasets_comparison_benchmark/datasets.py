"""Кастомные Dataset для классификации и сегментации"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import CFG
from .transforms import get_segmentation_transforms


class SubsetWithTransform(Dataset):
    """Обёртка над Subset: применяет torchvision-transform к изображению после выборки индекса."""

    def __init__(self, subset: Dataset, transform: Any = None) -> None:
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return len(self.subset)


class OxfordPetSegmentation(Dataset):
    """Oxford-IIIT Pet: бинарная маска переднего плана (trimap == 1), train/val из trainval.txt."""

    def __init__(self, root_dir: str, split: str = "train", image_size: int = 224) -> None:
        self.root = Path(root_dir)
        self.split = split
        self.image_size = image_size

        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "annotations" / "trimaps"
        self.split_file = self.root / "annotations" / "trainval.txt"

        self.image_ids = self._load_split()
        self.transform = get_segmentation_transforms(image_size, is_training=(split == "train"))

    def _load_split(self) -> List[str]:
        """Список ID изображений для split; при отсутствии trainval.txt — случайное разбиение."""
        if not self.split_file.exists():
            ids = [f.stem for f in self.images_dir.glob("*.jpg")]
            rng = np.random.RandomState(CFG.training.seed)
            rng.shuffle(ids)
            n = len(ids)
            if self.split == "train":
                return ids[: int(0.8 * n)]
            if self.split == "val":
                return ids[int(0.8 * n) : int(0.9 * n)]
            return ids[int(0.9 * n) :]

        trainval_ids = []
        with open(self.split_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    trainval_ids.append(parts[0])

        rng = np.random.RandomState(CFG.training.seed)
        order = rng.permutation(len(trainval_ids))
        n_train = int(0.8 * len(trainval_ids))

        if self.split == "train":
            return [trainval_ids[i] for i in order[:n_train]]
        if self.split == "val":
            return [trainval_ids[i] for i in order[n_train:]]
        return trainval_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        image_id = self.image_ids[idx]
        image = np.array(Image.open(self.images_dir / f"{image_id}.jpg").convert("RGB"))
        mask = np.array(Image.open(self.masks_dir / f"{image_id}.png"))

        mask = (mask == 1).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        augmented = self.transform(image=image, mask=mask)
        m = augmented["mask"]
        if isinstance(m, torch.Tensor):
            if m.dim() == 3 and m.shape[-1] == 1:
                m = m.permute(2, 0, 1).contiguous()
            elif m.dim() == 2:
                m = m.unsqueeze(0)
        return augmented["image"], m, image_id
