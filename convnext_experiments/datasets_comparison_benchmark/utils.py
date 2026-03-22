"""Вспомогательные функции: сид и вывод информации об устройстве."""

import random

import numpy as np
import torch

from .config import CFG


def set_seed(seed: int) -> None:
    """Фиксация генераторов для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_device_info() -> None:
    """Печать GPU/CPU"""
    print(f"Device: {CFG.hardware.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
