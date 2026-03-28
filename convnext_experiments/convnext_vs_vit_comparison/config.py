"""Конфигурация эксперимента сравнения ConvNeXt и ViT."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch


@dataclass
class PathConfig:
    """Каталоги данных, корня проекта и вывода артефактов."""

    experiment_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
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
    scheduler_eta_min: float = 1e-6
    seed: int = 42


@dataclass
class ModelSpec:
    """Спецификация одной модели для сравнения."""

    name: str
    timm_name: str
    pretrained: bool = True
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0


@dataclass
class ExperimentConfig:
    """Параметры эксперимента: модели для сравнения и датасеты."""

    batch_size: int = 32
    dropout: float = 0.2

    models: List[ModelSpec] = field(
        default_factory=lambda: [
            ModelSpec(
                name="ConvNeXt-Tiny",
                timm_name="convnext_tiny",
                pretrained=True,
                drop_path_rate=0.1,
            ),
            ModelSpec(
                name="ViT-Small",
                timm_name="vit_small_patch16_224",
                pretrained=True,
                drop_rate=0.1,
            ),
        ]
    )

    datasets: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "Flowers-5", "path": "Flowers_Dataset", "epochs": 10},
            {"name": "ImageNet-Mini", "path": "ImageNet-Mini/images", "epochs": 8},
        ]
    )


@dataclass
class Config:
    """Сводная конфигурация эксперимента ConvNeXt vs ViT."""

    paths: PathConfig = field(default_factory=PathConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


CFG = Config()
