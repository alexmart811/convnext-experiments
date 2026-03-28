"""Точка входа: сравнение ConvNeXt и ViT на нескольких датасетах, MLflow, итоговый бенчмарк."""

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

from convnext_experiments.convnext_vs_vit_comparison.benchmark import create_benchmark
from convnext_experiments.convnext_vs_vit_comparison.config import CFG
from convnext_experiments.convnext_vs_vit_comparison.data_loading import ensure_dataset, load_dataset
from convnext_experiments.convnext_vs_vit_comparison.training import train_and_evaluate
from convnext_experiments.convnext_vs_vit_comparison.utils import print_device_info, set_seed


def main(overrides: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Для каждого датасета обучает каждую модель из конфига,
    собирает результаты и строит сводный бенчмарк.
    """
    set_seed(CFG.training.seed)
    print_device_info()

    CFG.paths.output_dir.mkdir(parents=True, exist_ok=True)
    mlruns_uri = (CFG.paths.output_dir / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(mlruns_uri)
    mlflow.set_experiment("ConvNeXt-vs-ViT")

    results_list: List[Dict[str, Any]] = []

    for ds_cfg in CFG.experiment.datasets:
        raw_path = str(CFG.paths.data_root / ds_cfg["path"])
        data_path = ensure_dataset(raw_path, ds_cfg["path"])

        if not os.path.exists(data_path):
            print(f"Dataset not found: {data_path}")
            continue

        print(f"\nЗагрузка датасета {ds_cfg['name']}...")
        train_ds, val_ds, test_ds, classes = load_dataset(
            data_path, CFG.hardware.image_size, ds_cfg["name"]
        )
        num_epochs = int(ds_cfg["epochs"])

        for model_spec in CFG.experiment.models:
            result = train_and_evaluate(
                model_spec=model_spec,
                dataset_name=ds_cfg["name"],
                train_dataset=train_ds,
                val_dataset=val_ds,
                test_dataset=test_ds,
                class_names=classes,
                output_dir=CFG.paths.output_dir,
                num_epochs=num_epochs,
            )
            result["num_epochs"] = num_epochs
            results_list.append(result)

    if results_list:
        create_benchmark(
            results_list,
            str(CFG.paths.output_dir / "convnext_vs_vit_benchmark.png"),
        )

    return results_list


if __name__ == "__main__":
    main()
