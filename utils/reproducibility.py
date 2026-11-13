"""Shared helpers for reproducible experiments.

All training, analysis and figure notebooks should call ``set_seed(42)`` at
startup. This resets:

* the Python ``random`` module,
* the NumPy global random state,
* the PyTorch CPU/CUDA random state,
* cuDNN's deterministic algorithm selection

to the same starting point.

Note: cuDNN deterministic mode can slow down some convolution layers; SCORENet
uses only fully connected layers, so it has no measurable cost in this project.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> int:
    """Seed the Python, NumPy and PyTorch random number generators.

    Args:
        seed: Seed value. The default 42 is the value used to produce all
            checkpoints in the repository.

    Returns:
        The seed value (for chained calls).

    Note:
        ``PYTHONHASHSEED`` is read when the interpreter starts; setting it here
        has no effect on the current process. For full reproducibility, set the
        environment variable before starting the process::

            PYTHONHASHSEED=42 python ...
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducible (deterministic) cuDNN and matmul behavior.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        # Reject non-deterministic CUDA kernels for some operators.
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):
        pass

    return seed
