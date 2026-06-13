"""Shared utilities."""

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and (if available) PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic GPU kernels for reproducibility (at a small speed cost).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
