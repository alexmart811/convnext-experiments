"""Аугментации для классификации и сегментации"""

from __future__ import annotations

from typing import Any

from torchvision.transforms import Compose


def get_classification_transforms(image_size: int = 224, is_training: bool = True) -> Compose:
    """Нормализованные аугментации ImageNet для train/val-тест пайплайна классификации."""
    from torchvision import transforms

    if is_training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_segmentation_transforms(image_size: int = 224, is_training: bool = True) -> Any:
    """Albumentations Compose: изображение и маска"""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError as e:
        raise ImportError("Install albumentations: uv add albumentations") from e

    if is_training:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            additional_targets={"mask": "mask"},
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        additional_targets={"mask": "mask"},
    )
