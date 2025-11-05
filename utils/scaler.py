import numpy as np

from sklearn.preprocessing import StandardScaler


def standard_scaling(x: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()

    return scaler.fit_transform(x)
