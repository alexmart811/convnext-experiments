"""Точка входа: сегментация и классификация по CFG, MLflow, итоговый бенчмарк"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ is None or __package__ == "":
    _root = Path(__file__).resolve().parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import mlflow

from convnext_experiments.datasets_comparison_benchmark.benchmark import create_benchmark
from convnext_experiments.datasets_comparison_benchmark.config import CFG
from convnext_experiments.datasets_comparison_benchmark.data_loading import (
    create_segmentation_dataloaders,
    load_classification_dataset,
)
from convnext_experiments.datasets_comparison_benchmark.training import (
    train_classification,
    train_segmentation,
)
from convnext_experiments.datasets_comparison_benchmark.utils import print_device_info, set_seed


def main(overrides: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Запуск всех датасетов из конфига.
    """
    set_seed(CFG.training.seed)
    print_device_info()

    CFG.paths.output_dir.mkdir(parents=True, exist_ok=True)
    mlruns_uri = (CFG.paths.output_dir / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(mlruns_uri)
    mlflow.set_experiment("ConvNeXt-Benchmark")

    results_list = []

    for ds_cfg in CFG.segmentation.datasets:
        data_path = str(CFG.paths.data_root / ds_cfg["path"])
        if os.path.exists(data_path):
            num_epochs = int(ds_cfg["epochs"])
            train_loader, val_loader, dataset = create_segmentation_dataloaders(
                data_path, CFG.hardware.image_size, CFG.segmentation.batch_size
            )
            results = train_segmentation(
                ds_cfg["name"],
                train_loader,
                val_loader,
                dataset,
                CFG.paths.output_dir,
                num_epochs=num_epochs,
            )
            results["num_epochs"] = num_epochs
            results_list.append(results)
        else:
            print(f"Dataset not found: {data_path}")

    for ds_cfg in CFG.classification.datasets:
        data_path = str(CFG.paths.data_root / ds_cfg["path"])
        if os.path.exists(data_path):
            train_ds, val_ds, test_ds, classes = load_classification_dataset(
                data_path, CFG.hardware.image_size, ds_cfg["name"]
            )
            num_epochs = int(ds_cfg["epochs"])
            results = train_classification(
                ds_cfg["name"],
                train_ds,
                val_ds,
                test_ds,
                classes,
                CFG.paths.output_dir,
                num_epochs=num_epochs,
            )
            results["num_epochs"] = num_epochs
            results_list.append(results)
        else:
            print(f"Dataset not found: {data_path}")

    if results_list:
        create_benchmark(results_list, str(CFG.paths.output_dir / "benchmark_comparison.png"))

    return results_list


if __name__ == "__main__":
    main()
