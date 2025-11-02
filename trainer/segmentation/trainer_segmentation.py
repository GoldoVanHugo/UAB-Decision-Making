import abc
import numpy as np

from io_data import readNifty

from ..trainer_base import TrainerBase
from constants import SEED


class TrainerSegmentation(TrainerBase):
    @abc.abstractmethod
    def _get_image(self, image_path: str) -> np.ndarray:
        return super()._get_image(image_path=image_path)

    def _get_mask(self, mask_path: str) -> np.ndarray:
        mask, _ = readNifty(filePath=mask_path)

        # flatten data
        return mask.ravel()

    def _get_dataset(self, image_paths: list[str], mask_paths: list[str], voxel_ratio: float = None) -> tuple[
                     np.ndarray, np.ndarray]:
        X, Y = [], []

        for image_path, mask_path in zip(image_paths, mask_paths):
            x = self._get_image(image_path=image_path)
            y = self._get_mask(mask_path=mask_path)

            if voxel_ratio is not None:
                if self.set_seed:
                    np.random.seed(seed=SEED)

                unique_labels = np.unique(y)
                n_total = len(y)
                n_sample_total = int(voxel_ratio * n_total)

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

        return self._scale_x(x=X), Y

    @abc.abstractmethod
    def save_model(self):
        super().save_model()

    @abc.abstractmethod
    def load_model(self, model_path: str):
        super().load_model(model_path=model_path)

    @abc.abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray):
        super().train(x=x, y=y)

    @abc.abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        return super().predict(x=x)
    