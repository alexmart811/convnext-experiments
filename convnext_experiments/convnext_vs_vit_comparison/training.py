"""Циклы обучения и оценки для классификации с логированием в MLflow."""

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset

from .config import CFG, ModelSpec
from .metrics import MetricsTracker, compute_classification_metrics, plot_confusion_matrix
from .models import create_model


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> Tuple[float, float]:
    """Одна эпоха обучения: средний loss и accuracy."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    scaler = amp.GradScaler() if CFG.hardware.use_amp and device.type == "cuda" else None

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def _evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, List[int], List[int]]:
    """Оценка на выборке: loss, accuracy, предсказания и метки."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return running_loss / total, correct / total, all_preds, all_labels


def train_and_evaluate(
    model_spec: ModelSpec,
    dataset_name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    class_names: List[str],
    output_dir: Path,
    num_epochs: int,
) -> Dict[str, Any]:
    """
    Полный цикл: обучение, выбор лучших весов по val accuracy,
    оценка на тесте, логирование артефактов в MLflow.
    """
    print(f"\n{'=' * 55}")
    print(f"  {model_spec.name} на {dataset_name}")
    print(f"{'=' * 55}")

    device = CFG.hardware.device
    cfg = CFG.experiment

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=CFG.hardware.num_workers,
        pin_memory=CFG.hardware.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=CFG.hardware.num_workers,
        pin_memory=CFG.hardware.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=CFG.hardware.num_workers,
        pin_memory=CFG.hardware.pin_memory,
    )

    model = create_model(model_spec, num_classes=len(class_names)).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.training.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG.training.learning_rate,
        weight_decay=CFG.training.weight_decay,
        betas=CFG.training.betas,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=CFG.training.scheduler_eta_min
    )

    tracker = MetricsTracker()
    best_val_acc, best_weights = 0.0, copy.deepcopy(model.state_dict())

    run_name = f"{dataset_name}_{model_spec.name.replace(' ', '_')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "dataset": dataset_name,
                "model": model_spec.name,
                "timm_name": model_spec.timm_name,
                "num_classes": len(class_names),
                "num_params": num_params,
                "num_epochs": num_epochs,
                "batch_size": cfg.batch_size,
                "learning_rate": CFG.training.learning_rate,
                "image_size": CFG.hardware.image_size,
                "pretrained": model_spec.pretrained,
            }
        )

        for epoch in range(num_epochs):
            t0 = time.time()

            train_loss, train_acc = _train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, device
            )
            val_loss, val_acc, _, _ = _evaluate(model, val_loader, criterion, device)

            epoch_time = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]
            tracker.update(train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "learning_rate": current_lr,
                    "epoch_time_s": epoch_time,
                },
                step=epoch,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())

            print(
                f"  Epoch {epoch + 1}/{num_epochs} | "
                f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

        model.load_state_dict(best_weights)
        test_loss, test_acc, test_preds, test_labels = _evaluate(
            model, test_loader, criterion, device
        )
        cls_metrics = compute_classification_metrics(
            np.array(test_preds), np.array(test_labels), len(class_names)
        )

        avg_epoch_time = sum(tracker.epoch_times) / len(tracker.epoch_times)

        mlflow.log_metrics(
            {
                "test_loss": test_loss,
                "test_acc": test_acc,
                "best_val_acc": best_val_acc,
                "precision": cls_metrics["precision"],
                "recall": cls_metrics["recall"],
                "f1_score": cls_metrics["f1"],
                "avg_epoch_time_s": avg_epoch_time,
            }
        )

        cm_path = plot_confusion_matrix(
            cls_metrics["confusion_matrix"],
            class_names,
            str(output_dir / f"cm_{dataset_name}_{model_spec.name.replace(' ', '_')}.png"),
        )
        mlflow.log_artifact(cm_path)
        mlflow.pytorch.log_model(model, "model")

        print(
            f"  Test: Acc={test_acc:.4f}, P={cls_metrics['precision']:.4f}, "
            f"R={cls_metrics['recall']:.4f}, F1={cls_metrics['f1']:.4f}, "
            f"Avg epoch: {avg_epoch_time:.1f}s"
        )

    return {
        "dataset": dataset_name,
        "model": model_spec.name,
        "num_params": num_params,
        "num_classes": len(class_names),
        "num_train": len(train_dataset),
        "num_test": len(test_dataset),
        "test_acc": test_acc,
        "best_val_acc": best_val_acc,
        "precision": cls_metrics["precision"],
        "recall": cls_metrics["recall"],
        "f1": cls_metrics["f1"],
        "avg_epoch_time": avg_epoch_time,
        "metrics_tracker": tracker,
        "confusion_matrix": cls_metrics["confusion_matrix"],
        "class_names": class_names,
    }
