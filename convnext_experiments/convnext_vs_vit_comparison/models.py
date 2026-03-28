"""Создание моделей для сравнения ConvNeXt и ViT через timm."""

import timm
import torch.nn as nn

from .config import ModelSpec


def create_model(spec: ModelSpec, num_classes: int) -> nn.Module:
    """
    Универсальная фабрика: создаёт модель по спецификации из конфига.
    Поддерживает ConvNeXt (drop_path_rate) и ViT (drop_rate).
    """
    model = timm.create_model(
        spec.timm_name,
        pretrained=spec.pretrained,
        num_classes=num_classes,
        drop_rate=spec.drop_rate,
        drop_path_rate=spec.drop_path_rate,
    )

    num_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {spec.name}: {num_params:,} params ({trainable:,} trainable)")

    return model
