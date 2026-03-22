# Сравнение метрик на различных датасетах (datasets_comparison_benchmark)

**Автор:** Мартынов Александр.

## Суть эксперимента

Обучение модели **convnext_tiny** на нескольких наборах данных со сравнением метрик.

## Датасеты

| Название | Задача |
|----------|--------|
| [Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset) | Классификация |
| [Mini-ImageNet](https://www.kaggle.com/datasets/deeptrial/miniimagenet) | Классификация |
| [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) | Сегментация |

## Структура файлов

```text
datasets_comparison_benchmark/
├── README.md                 
├── __init__.py               
├── main.py                   # запуск пайплайна, MLflow, сводный бенчмарк
├── config.py                 # общий конфигурационный файл с константами 
├── utils.py                  # фиксация seed, GPU/CPU
├── transforms.py             # аугментации классификации и сегментации
├── datasets.py               # инициализация классов датасетов
├── data_loading.py           # ImageFolder, сплиты, DataLoader сегментации
├── models.py                 # ConvNeXt (cls), TimmSegmentationModel (seg)
├── metrics.py                # метрики, confusion matrix, графики сегментации
├── training.py               # эпохи обучения, MLflow-логирование
├── benchmark.py              # Сбор графиков с метриками
└── runs/                     # хранилище логов обучения и графиков с метриками
```

## Запуск

Из корня репозитория:

```bash
uv run python convnext_experiments/datasets_comparison_benchmark/main.py
```
