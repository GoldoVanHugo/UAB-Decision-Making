import numpy as np

from constants import SEED


def shuffle(x: np.ndarray, y: np.ndarray, set_seed: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if set_seed:
        np.random.seed(seed=SEED)

    indices = np.arange(len(x))
    np.random.shuffle(indices)

    return x[indices], y[indices]
