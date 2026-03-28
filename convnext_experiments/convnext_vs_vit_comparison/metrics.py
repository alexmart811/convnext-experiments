"""Метрики классификации и визуализация матрицы ошибок."""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class MetricsTracker:
    """Хранение метрик по эпохам для последующих графиков сходимости."""

    def __init__(self) -> None:
        self.train_loss: List[float] = []
        self.train_acc: List[float] = []
        self.val_loss: List[float] = []
        self.val_acc: List[float] = []
        self.learning_rates: List[float] = []
        self.epoch_times: List[float] = []

    def update(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        epoch_time: float = 0.0,
    ) -> None:
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)


def compute_classification_metrics(
    preds: np.ndarray, labels: np.ndarray, num_classes: int
) -> Dict[str, float | np.ndarray]:
    """Precision / Recall / F1 (weighted) и матрица ошибок."""
    label_index = np.arange(num_classes)

    precision = precision_score(
        labels, preds, average="weighted", labels=label_index, zero_division=0
    )
    recall = recall_score(labels, preds, average="weighted", labels=label_index, zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", labels=label_index, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=label_index)

    return {"precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str) -> str:
    """Сохранение heatmap матрицы ошибок в PNG."""
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
