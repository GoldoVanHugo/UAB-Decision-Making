import numpy as np

from .dataloader_base import DataloaderBase

from utils import standard_scaling
from constants import (
    SEED,
    VOXEL_TRAIN_RATIO
)


class DataloaderSegmentation(DataloaderBase):
    def _get_dataset(self, image_paths: list[str], mask_paths: list[str], train: bool = True) -> tuple[
                     np.ndarray, np.ndarray]:
        X, Y = [], []

        for image_path, mask_path in zip(image_paths, mask_paths):
            x = self._get_image(image_path=image_path)
            y = self._get_mask(mask_path=mask_path)

            if not self.use_all_voxels and train:
                if self.set_seed:
                    np.random.seed(seed=SEED)

                unique_labels = np.unique(y)
                n_total = len(y)
                n_sample_total = int(VOXEL_TRAIN_RATIO * n_total)

                n_per_label = n_sample_total // len(unique_labels)

                sample_idx = []

                for label in unique_labels:
                    label_indices = np.where(y == label)[0]
                    n_select = min(n_per_label, len(label_indices))
                    chosen = np.random.choice(label_indices, size=n_select, replace=False)
                    sample_idx.append(chosen)

                sample_idx = np.concatenate(sample_idx)
                x = x[sample_idx]
                y = y[sample_idx]

            X.append(x)
            Y.append(y)

        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)

        return standard_scaling(x=X), Y
