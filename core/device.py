# core/device.py
"""
Device management — single source of truth for compute device.
No model logic. No I/O.
"""
from __future__ import annotations

import torch


def get_device(prefer: str = "cuda") -> str:
    """Return best available device string."""
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def setup_cuda() -> None:
    """One-time CUDA optimizations. Call once at startup."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if torch.__version__ >= "2.0":
        torch.set_float32_matmul_precision("high")


def setup_seed(seed: int = 42) -> None:
    """Deterministic seeding for reproducibility."""
    import random
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
