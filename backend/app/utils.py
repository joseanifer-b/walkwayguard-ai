from __future__ import annotations

import random
import time
import uuid

import numpy as np


def get_device():
    """Device selection. Override with: export WALKWAYGUARD_DEVICE=cpu|mps|cuda"""
    import os
    import torch

    forced = os.environ.get("WALKWAYGUARD_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")  # MPS disabled by default: it crashes on some macOS/PyTorch combinations


def seed_everything(seed: int = 42) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
