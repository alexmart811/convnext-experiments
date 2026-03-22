"""Модели для классификации и сегментации"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny


def create_classification_model(
    num_classes: int, pretrained: bool = True, dropout: float = 0.2
) -> nn.Module:
    """Создание модели классификации"""
    model = convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
    num_features = model.classifier[2].in_features

    model.classifier = nn.Sequential(
        nn.Flatten(1),
        nn.LayerNorm(num_features),
        nn.Dropout(dropout),
        nn.Linear(num_features, num_classes),
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Classification model: ConvNeXt-Tiny ({num_params:,} parameters)")
    return model


class TimmSegmentationModel(nn.Module):
    """
    Сегментация: энкодер timm (features_only) и свёрточный декодер
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int = 1,
        pretrained: bool = True,
        decoder_channels: int = 256,
    ) -> None:
        super().__init__()

        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),
        )

        with torch.no_grad():
            feat = self.encoder(torch.zeros(1, 3, 224, 224))
            in_ch = feat[-1].shape[1]

        self.seg_head = nn.Sequential(
            nn.Conv2d(in_ch, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)[-1]
        logits = self.seg_head(feat)
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)


def create_segmentation_model(
    backbone: str = "convnext_tiny",
    num_classes: int = 1,
    pretrained: bool = True,
    decoder_channels: int = 256,
) -> TimmSegmentationModel:
    """Создание модели классификации"""
    model = TimmSegmentationModel(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        decoder_channels=decoder_channels,
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"Segmentation model: timm[{backbone}] + decoder ({n:,} params)")
    return model
