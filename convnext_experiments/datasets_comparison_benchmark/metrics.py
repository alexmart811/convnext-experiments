"""Метрики и визуализации: классификация (матрица ошибок) и сегментация (IoU, Dice)."""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class MetricsTracker:
    """Хранение по эпохам train/val loss, accuracy и learning rate для графиков."""

    def __init__(self) -> None:
        self.train_loss: List[float] = []
        self.train_acc: List[float] = []
        self.val_loss: List[float] = []
        self.val_acc: List[float] = []
        self.learning_rates: List[float] = []

    def update(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
    ) -> None:
        """Добавление значения за одну эпоху."""
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.learning_rates.append(lr)


def compute_classification_metrics(
    preds: np.ndarray, labels: np.ndarray, num_classes: int
) -> Dict[str, np.ndarray | float]:
    """Precision/Recall/F1 (weighted) и матрица ошибок по всем классам."""
    label_index = np.arange(num_classes)

    precision = precision_score(
        labels, preds, average="weighted", labels=label_index, zero_division=0
    )
    recall = recall_score(labels, preds, average="weighted", labels=label_index, zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", labels=label_index, zero_division=0)

    cm = confusion_matrix(labels, preds, labels=label_index)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str) -> str:
    """Сохранение heatmap матрицы ошибок"""
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    tick_labels = (
        list(class_names)
        if n <= 48
        else [class_names[i] if i % max(1, n // 24) == 0 else "" for i in range(n)]
    )

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        xlabel="Predicted",
        ylabel="True",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    if n <= 64:
        thresh = cm.max() / 2.0 if cm.size else 0.0
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def compute_segmentation_metrics(
    preds: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
    *,
    from_logits: bool = True,
) -> Dict[str, float]:
    """
    IoU, Dice и pixel accuracy для бинарной маски.
    """
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(masks, torch.Tensor):
        masks = torch.tensor(masks)

    if preds.dim() == 3:
        preds = preds.unsqueeze(0)
    if masks.dim() == 3:
        masks = masks.unsqueeze(0)

    probs = torch.sigmoid(preds) if from_logits else preds
    masks = masks.float().clamp(0.0, 1.0)

    eps = 1e-6
    preds_binary = (probs > threshold).float()

    intersection = (preds_binary * masks).sum(dim=(1, 2, 3))
    union = preds_binary.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
    iou = ((intersection + eps) / (union + eps)).mean().item()

    dice = (
        (
            (2.0 * intersection + eps)
            / (preds_binary.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + eps)
        )
        .mean()
        .item()
    )

    pixel_acc = (preds_binary == masks).float().mean().item()

    return {"iou": iou, "dice": dice, "pixel_accuracy": pixel_acc}


def plot_segmentation_results(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    image_ids: List[str],
    save_path: str,
    num_samples: int = 4,
) -> str:
    """Сетка изображение / GT / предсказание для первых num_samples примеров."""
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    probs = torch.sigmoid(preds) if preds.shape[1] == 1 else preds
    preds_binary = (probs > 0.5).float()

    for i in range(min(num_samples, len(images))):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        mask = masks[i].squeeze().cpu().numpy()
        pred = preds_binary[i].squeeze().cpu().numpy()

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"{image_ids[i]}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path
