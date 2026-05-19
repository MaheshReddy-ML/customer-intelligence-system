import random

import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments.

    Args:
        seed: Integer seed value.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
