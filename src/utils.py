
"""
Utility helpers for the project.

Provides a deterministic global seed setter so runs are reproducible.
"""

import os
import random
import numpy as np
import pickle


def set_global_seed(seed: int) -> None:
    # Set seeds for Python, numpy and environment to make runs deterministic.

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)