# Сравнение ConvNeXt и ViT (convnext_vs_vit_comparison)

**Автор:** Кадиев Малик Нурдинович.

## Суть эксперимента

Сравнение свёрточной архитектуры **ConvNeXt-Tiny** и трансформерной **ViT-Small** на задаче классификации изображений с использованием предобученных на ImageNet весов.

Цель — оценить разницу в качестве, скорости сходимости и времени обучения между современными CNN и Vision Transformer на одних и тех же данных и в одинаковых условиях.

## Датасеты

| Название | Задача | Эпох |
|----------|--------|------|
| [Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset) | Классификация | 10 |
| [Mini-ImageNet](https://www.kaggle.com/datasets/deeptrial/miniimagenet) | Классификация | 8 |

(Датасеты скачиваются автоматически при первом запуске через `kagglehub`)

## Сравниваемые модели

| Модель | Архитектура | Параметры (≈) | Источник |
|--------|-------------|---------------|----------|
| ConvNeXt-Tiny | CNN (свёрточная сеть) | 28.6M | [timm](https://github.com/huggingface/pytorch-image-models) |
| ViT-Small/16 | Transformer (патчи 16×16) | 22.1M | [timm](https://github.com/huggingface/pytorch-image-models) |

## Отслеживаемые метрики

- **Accuracy** (test) — точность на тестовой выборке
- **Precision / Recall / F1** (weighted) — взвешенные по классам
- **Среднее время эпохи** — сравнение скорости обучения
- **Кривые сходимости** — val accuracy по эпохам
- **Confusion matrix** — для каждой пары модель + датасет

Все метрики логируются в **MLflow**.

## Структура файлов

```text
convnext_vs_vit_comparison/
├── README.md             # описание эксперимента
├── __init__.py
├── main.py               # точка входа: запуск всех экспериментов, MLflow, бенчмарк
├── config.py             # конфигурация: модели, датасеты, гиперпараметры
├── utils.py              # фиксация seed, информация об устройстве
├── transforms.py         # аугментации для train и val/test
├── data_loading.py       # загрузка ImageFolder-датасетов, авто-скачивание через kagglehub
├── models.py             # создание моделей через timm (ConvNeXt-Tiny, ViT-Small)
├── metrics.py            # MetricsTracker, precision/recall/F1, confusion matrix
├── training.py           # циклы обучения и оценки, логирование в MLflow
├── benchmark.py          # сводные графики сравнения моделей
└── runs/                 # артефакты обучения (создаётся автоматически)
```

## Запуск

Из корня репозитория:

```bash
uv run python -m convnext_experiments.convnext_vs_vit_comparison.main
```

Для обучения на GPU необходимо установить PyTorch с поддержкой CUDA:

```bash
uv sync --index-url https://download.pytorch.org/whl/cu126
```
