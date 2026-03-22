"""Конфигурация пайплайна."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch


@dataclass
class PathConfig:
    """Каталоги данных, корня проекта и вывода артефактов."""

    benchmark_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )
    data_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data"
    )
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "runs")

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class HardwareConfig:
    """Устройство, размер входа и параметры DataLoader."""

    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    image_size: int = 224
    num_workers: int = 2
    pin_memory: bool = True
    use_amp: bool = True


@dataclass
class TrainingConfig:
    """Оптимизатор, планировщик LR и воспроизводимость."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    label_smoothing: float = 0.1
    scheduler_t_max: int = 20
    scheduler_eta_min: float = 1e-6
    seed: int = 42


@dataclass
class ClassificationConfig:
    """Классификация ConvNeXt"""

    batch_size: int = 32
    dropout: float = 0.2
    pretrained: bool = True
    model_name: str = "convnext_tiny"

    datasets: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "Flowers-5", "path": "Flowers_Dataset", "epochs": 10},
            {"name": "ImageNet-Mini", "path": "ImageNet-Mini/images", "epochs": 8},
        ]
    )


@dataclass
class SegmentationConfig:
    """Сегментация ConvNeXt"""

    batch_size: int = 16
    pretrained: bool = True
    backbone: str = "convnext_tiny"
    decoder_channels: int = 256
    num_classes: int = 1
    threshold: float = 0.5

    datasets: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "Oxford-Pet-Seg", "path": "The_Oxford_Pet_Dataset", "epochs": 5},
        ]
    )


@dataclass
class Config:
    """Сводная конфигурация эксперимент"""

    paths: PathConfig = field(default_factory=PathConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)

    @classmethod
    def from_dict(cls, overrides: Dict[str, Any]) -> "Config":
        """Сборка конфига"""
        config = cls()
        if "learning_rate" in overrides:
            config.training.learning_rate = overrides["learning_rate"]
        if "image_size" in overrides:
            config.hardware.image_size = overrides["image_size"]
        return config


CFG = Config()
