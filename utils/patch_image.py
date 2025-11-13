import numpy as np


def patching(image: np.ndarray, size_3d: int) -> np.ndarray:
    pad_image = np.pad(
        array=image,
        pad_width=size_3d // 2,
        mode="reflect",
    )
    patch_image = np.lib.stride_tricks.sliding_window_view(pad_image, window_shape=(size_3d, size_3d))

    return np.expand_dims(patch_image, axis=2)