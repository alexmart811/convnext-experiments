"""Сводная визуализация результатов: ConvNeXt vs ViT по датасетам."""

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def create_benchmark(results_list: List[Dict[str, Any]], save_path: str) -> None:
    """
    Построение 2x2 сетки графиков:
      [0,0] — Test Accuracy по моделям и датасетам
      [0,1] — Precision / Recall / F1
      [1,0] — Val Accuracy по эпохам (кривые сходимости)
      [1,1] — Таблица: параметры, среднее время эпохи, throughput
    """
    datasets_seen: List[str] = []
    for r in results_list:
        if r["dataset"] not in datasets_seen:
            datasets_seen.append(r["dataset"])

    models_seen: List[str] = []
    for r in results_list:
        if r["model"] not in models_seen:
            models_seen.append(r["model"])

    model_colors = {
        models_seen[i]: c
        for i, c in zip(range(len(models_seen)), ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"])
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("ConvNeXt vs ViT — Benchmark", fontsize=16, fontweight="bold", y=0.98)

    # --- [0,0] Test Accuracy ---
    ax = axes[0, 0]
    x = np.arange(len(datasets_seen))
    width = 0.8 / max(len(models_seen), 1)
    for i, model_name in enumerate(models_seen):
        accs = []
        for ds in datasets_seen:
            r = next((r for r in results_list if r["dataset"] == ds and r["model"] == model_name), None)
            accs.append(r["test_acc"] * 100 if r else 0)
        offset = (i - (len(models_seen) - 1) / 2) * width
        ax.bar(x + offset, accs, width * 0.9, label=model_name, color=model_colors[model_name], alpha=0.85)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_seen, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # --- [0,1] Precision / Recall / F1 ---
    ax = axes[0, 1]
    metric_names = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1"]
    group_labels = []
    group_values = {m: [] for m in metric_names}

    for ds in datasets_seen:
        for model_name in models_seen:
            r = next((r for r in results_list if r["dataset"] == ds and r["model"] == model_name), None)
            group_labels.append(f"{ds}\n{model_name}")
            for m in metric_names:
                group_values[m].append(r[m] * 100 if r else 0)

    x2 = np.arange(len(group_labels))
    w2 = 0.25
    for j, (m, label) in enumerate(zip(metric_names, metric_labels)):
        ax.bar(x2 + (j - 1) * w2, group_values[m], w2, label=label, alpha=0.8)
    ax.set_ylabel("Score (%)")
    ax.set_title("Precision / Recall / F1")
    ax.set_xticks(x2)
    ax.set_xticklabels(group_labels, fontsize=7, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # --- [1,0] Val Accuracy по эпохам ---
    ax = axes[1, 0]
    line_styles = ["-", "--", "-.", ":"]
    for i, r in enumerate(results_list):
        vals = [v * 100 for v in r["metrics_tracker"].val_acc]
        style = line_styles[i % len(line_styles)]
        ax.plot(
            range(1, len(vals) + 1),
            vals,
            style,
            label=f"{r['model']} / {r['dataset']}",
            color=model_colors.get(r["model"], "#333"),
            linewidth=2,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Accuracy (%)")
    ax.set_title("Convergence: Val Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- [1,1] Сводная таблица ---
    ax = axes[1, 1]
    ax.axis("off")
    table_data = []
    for r in results_list:
        table_data.append(
            [
                r["model"],
                r["dataset"],
                f"{r['num_params']:,}",
                f"{r['test_acc'] * 100:.2f}%",
                f"{r['f1'] * 100:.2f}%",
                f"{r['avg_epoch_time']:.1f}s",
            ]
        )

    table = ax.table(
        cellText=table_data,
        colLabels=["Model", "Dataset", "Params", "Accuracy", "F1", "Epoch Time"],
        loc="center",
        cellLoc="center",
        colWidths=[0.18, 0.18, 0.16, 0.14, 0.14, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    for col in range(6):
        table[(0, col)].set_facecolor("#34495e")
        table[(0, col)].set_text_props(color="white", weight="bold")
    ax.set_title("Summary", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nBenchmark saved: {save_path}")

    print("\nBenchmark Summary:")
    print("-" * 70)
    for r in results_list:
        print(
            f"  {r['model']:15s} | {r['dataset']:15s} | "
            f"Acc={r['test_acc'] * 100:.2f}% | F1={r['f1'] * 100:.2f}% | "
            f"Epoch={r['avg_epoch_time']:.1f}s"
        )
