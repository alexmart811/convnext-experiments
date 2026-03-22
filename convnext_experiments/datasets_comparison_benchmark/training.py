"""Циклы обучения и оценки для классификации и сегментации с логированием в MLflow."""

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

from .config import CFG
from .metrics import (
    MetricsTracker,
    compute_classification_metrics,
    compute_segmentation_metrics,
    plot_confusion_matrix,
    plot_segmentation_results,
)
from .models import create_classification_model, create_segmentation_model


def train_epoch_cls(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> Tuple[float, float]:
    """Одна эпоха классификации: средний loss и accuracy по батчам"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    scaler = amp.GradScaler() if CFG.hardware.use_amp else None

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if CFG.hardware.use_amp:
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


def evaluate_cls(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, List[Any], List[Any]]:
    """Оценка на выборке: loss, accuracy, списки предсказаний и меток для метрик на тесте"""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


def train_classification(
    dataset_name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    class_names: List[str],
    output_dir: Path,
    num_epochs: int,
) -> Dict[str, Any]:
    """
    Полный цикл обучения классификации: лучшие веса по val, тест и метрики, артефакты в MLflow.

    Возвращает словарь с метриками, трекером и матрицей ошибок для сводного бенчмарка.
    """
    print(f"\nTraining classification on {dataset_name}")
    print("-" * 50)

    cfg = CFG.classification
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

    model = create_classification_model(len(class_names), cfg.pretrained, cfg.dropout).to(
        CFG.hardware.device
    )
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

    metrics = MetricsTracker()
    best_val_acc, best_model_wts = 0.0, copy.deepcopy(model.state_dict())

    with mlflow.start_run(run_name=f"{dataset_name}_cls"):
        mlflow.log_params(
            {
                "dataset": dataset_name,
                "task": "classification",
                "num_classes": len(class_names),
                "num_epochs": num_epochs,
                "batch_size": cfg.batch_size,
                "learning_rate": CFG.training.learning_rate,
                "image_size": CFG.hardware.image_size,
                "model": "ConvNeXt-Tiny",
            }
        )

        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss, train_acc = train_epoch_cls(
                model, train_loader, criterion, optimizer, scheduler, CFG.hardware.device
            )
            val_loss, val_acc, _, _ = evaluate_cls(
                model, val_loader, criterion, CFG.hardware.device
            )

            current_lr = optimizer.param_groups[0]["lr"]
            metrics.update(train_loss, train_acc, val_loss, val_acc, current_lr)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "learning_rate": current_lr,
                },
                step=epoch,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            print(
                f"Epoch {epoch + 1}/{num_epochs} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
                f"Time: {time.time() - start_time:.1f}s"
            )

        model.load_state_dict(best_model_wts)
        test_loss, test_acc, test_preds, test_labels = evaluate_cls(
            model, test_loader, criterion, CFG.hardware.device
        )
        cls_metrics = compute_classification_metrics(
            np.array(test_preds), np.array(test_labels), len(class_names)
        )

        mlflow.log_metrics(
            {
                "test_loss": test_loss,
                "test_acc": test_acc,
                "best_val_acc": best_val_acc,
                "precision": cls_metrics["precision"],
                "recall": cls_metrics["recall"],
                "f1_score": cls_metrics["f1"],
            }
        )

        cm_path = plot_confusion_matrix(
            cls_metrics["confusion_matrix"],
            class_names,
            str(output_dir / f"confusion_matrix_{dataset_name}.png"),
        )
        mlflow.log_artifact(cm_path)
        mlflow.pytorch.log_model(model, "model")

        print(
            f"Test Results: Acc={test_acc:.4f}, Precision={cls_metrics['precision']:.4f}, "
            f"Recall={cls_metrics['recall']:.4f}, F1={cls_metrics['f1']:.4f}"
        )

        return {
            "dataset": dataset_name,
            "task": "classification",
            "num_classes": len(class_names),
            "num_train": len(train_dataset),
            "num_test": len(test_dataset),
            "test_acc": test_acc,
            "best_val_acc": best_val_acc,
            "precision": cls_metrics["precision"],
            "recall": cls_metrics["recall"],
            "f1": cls_metrics["f1"],
            "metrics_tracker": metrics,
            "confusion_matrix": cls_metrics["confusion_matrix"],
            "class_names": class_names,
        }


def train_epoch_seg(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> float:
    """Одна эпоха сегментации: средний loss BCE на логитах"""
    model.train()
    running_loss, total = 0.0, 0
    scaler = amp.GradScaler() if CFG.hardware.use_amp and device.type == "cuda" else None

    for images, masks, _ in dataloader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        if CFG.hardware.use_amp and device.type == "cuda":
            with amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    return running_loss / total


def evaluate_seg(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Средний loss по всем батчам, метрики сегментации на объединённых предсказаниях,
    тензоры изображений/масок/логитов и список id для визуализации.
    """
    model.eval()
    running_loss, total = 0.0, 0
    all_preds, all_masks, all_images, all_ids = [], [], [], []

    with torch.no_grad():
        for images, masks, image_ids in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)
            total += images.size(0)

            all_preds.append(outputs.cpu())
            all_masks.append(masks.cpu())
            all_images.append(images.cpu())
            all_ids.extend(image_ids)

    return (
        running_loss / total,
        compute_segmentation_metrics(
            torch.cat(all_preds, 0), torch.cat(all_masks, 0), CFG.segmentation.threshold
        ),
        torch.cat(all_images, 0),
        torch.cat(all_masks, 0),
        torch.cat(all_preds, 0),
        all_ids,
    )


def train_segmentation(
    dataset_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    _dataset: Dataset,
    output_dir: Path,
    num_epochs: int,
) -> Dict[str, Any]:
    """
    Обучение сегментации: лучшие веса по val IoU, итоговые метрики на val и визуализация в MLflow.
    """
    print(f"\nTraining segmentation on {dataset_name}")
    print("-" * 50)

    cfg = CFG.segmentation
    model = create_segmentation_model(
        backbone=cfg.backbone,
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained,
        decoder_channels=cfg.decoder_channels,
    ).to(CFG.hardware.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG.training.learning_rate,
        weight_decay=CFG.training.weight_decay,
        betas=CFG.training.betas,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=CFG.training.scheduler_eta_min
    )

    best_val_iou, best_model_wts = 0.0, copy.deepcopy(model.state_dict())

    with mlflow.start_run(run_name=f"{dataset_name}_seg"):
        mlflow.log_params(
            {
                "dataset": dataset_name,
                "task": "segmentation",
                "encoder": "timm",
                "backbone": cfg.backbone,
                "decoder_channels": cfg.decoder_channels,
                "num_epochs": num_epochs,
                "learning_rate": CFG.training.learning_rate,
            }
        )

        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss = train_epoch_seg(
                model, train_loader, criterion, optimizer, scheduler, CFG.hardware.device
            )
            train_eval_loss, train_metrics, _, _, _, _ = evaluate_seg(
                model, train_loader, criterion, CFG.hardware.device
            )
            val_loss, val_metrics, _, _, _, _ = evaluate_seg(
                model, val_loader, criterion, CFG.hardware.device
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_eval_loss": train_eval_loss,
                    "train_iou": train_metrics["iou"],
                    "train_dice": train_metrics["dice"],
                    "train_pixel_acc": train_metrics["pixel_accuracy"],
                    "val_loss": val_loss,
                    "val_iou": val_metrics["iou"],
                    "val_dice": val_metrics["dice"],
                    "val_pixel_acc": val_metrics["pixel_accuracy"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

            if val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                best_model_wts = copy.deepcopy(model.state_dict())

            print(
                f"Epoch {epoch + 1}/{num_epochs} | Train IoU: {train_metrics['iou']:.4f} | "
                f"Val IoU: {val_metrics['iou']:.4f} | Time: {time.time() - start_time:.1f}s"
            )

        model.load_state_dict(best_model_wts)
        _, test_metrics, test_images, test_masks, test_preds, test_ids = evaluate_seg(
            model, val_loader, criterion, CFG.hardware.device
        )

        mlflow.log_metrics(
            {
                "test_iou": test_metrics["iou"],
                "test_dice": test_metrics["dice"],
                "test_pixel_acc": test_metrics["pixel_accuracy"],
                "best_val_iou": best_val_iou,
            }
        )

        viz_path = plot_segmentation_results(
            test_images,
            test_masks,
            test_preds,
            test_ids,
            str(output_dir / f"seg_viz_{dataset_name}.png"),
        )
        mlflow.log_artifact(viz_path)
        mlflow.pytorch.log_model(model, "model")

        print(
            f"Test Results: IoU={test_metrics['iou']:.4f}, Dice={test_metrics['dice']:.4f}, "
            f"Pixel Acc={test_metrics['pixel_accuracy']:.4f}"
        )

        return {
            "dataset": dataset_name,
            "task": "segmentation",
            "iou": test_metrics["iou"],
            "dice": test_metrics["dice"],
            "pixel_accuracy": test_metrics["pixel_accuracy"],
            "best_val_iou": best_val_iou,
        }
