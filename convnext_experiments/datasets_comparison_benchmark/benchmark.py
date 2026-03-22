"""Визуализация результатов классификации и сегментации."""

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def create_benchmark(results_list: List[Dict[str, Any]], save_path: str) -> None:
    """
    Построение сетки графиков (accuracy, Precision/Recall/F1, val accuracy по эпохам, таблица IoU/Dice)
    и сохранение PNG.
    """
    cls_results = [r for r in results_list if r.get("task") == "classification"]
    seg_results = [r for r in results_list if r.get("task") == "segmentation"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    if cls_results:
        datasets = [r["dataset"] for r in cls_results]
        x = np.arange(len(datasets))

        axes[0, 0].bar(
            x, [r["test_acc"] for r in cls_results], color=colors[: len(datasets)], alpha=0.8
        )
        axes[0, 0].set_xlabel("Dataset")
        axes[0, 0].set_ylabel("Test Accuracy")
        axes[0, 0].set_title("Classification: Test Accuracy")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(datasets, rotation=15)
        axes[0, 0].grid(True, alpha=0.3, axis="y")

        width = 0.25
        axes[0, 1].bar(
            x - width,
            [r["precision"] for r in cls_results],
            width,
            label="Precision",
            color=colors[0],
            alpha=0.8,
        )
        axes[0, 1].bar(
            x,
            [r["recall"] for r in cls_results],
            width,
            label="Recall",
            color=colors[1],
            alpha=0.8,
        )
        axes[0, 1].bar(
            x + width,
            [r["f1"] for r in cls_results],
            width,
            label="F1",
            color=colors[2],
            alpha=0.8,
        )
        axes[0, 1].set_xlabel("Dataset")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].set_title("Classification: Precision/Recall/F1")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(datasets, rotation=15)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis="y")

        for i, r in enumerate(cls_results):
            axes[1, 0].plot(
                r["metrics_tracker"].val_acc,
                label=f"{r['dataset']}",
                color=colors[i % len(colors)],
                linewidth=2,
            )
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Validation Accuracy")
        axes[1, 0].set_title("Classification: Val Accuracy Over Epochs")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    if seg_results:
        axes[1, 1].axis("off")
        seg_data = [
            [
                r["dataset"],
                f"{r['iou'] * 100:.2f}%",
                f"{r['dice'] * 100:.2f}%",
                f"{r['pixel_accuracy'] * 100:.2f}%",
            ]
            for r in seg_results
        ]

        if seg_data:
            table = axes[1, 1].table(
                cellText=seg_data,
                colLabels=["Dataset", "IoU", "Dice", "Pixel Acc"],
                loc="center",
                cellLoc="center",
                colWidths=[0.3, 0.2, 0.2, 0.2],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.8)
            for i in range(len(seg_data) + 1):
                table[(i, 0)].set_facecolor("#3498db")
                table[(i, 0)].set_text_props(color="white", weight="bold")
            axes[1, 1].set_title("Segmentation Metrics", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Benchmark saved: {save_path}")

    print("\nBenchmark Summary:")
    print("-" * 50)
    for r in results_list:
        if r["task"] == "classification":
            print(f"{r['dataset']} (cls): Acc={r['test_acc']*100:.2f}%, F1={r['f1']*100:.2f}%")
        else:
            print(f"{r['dataset']} (seg): IoU={r['iou']*100:.2f}%, Dice={r['dice']*100:.2f}%")
