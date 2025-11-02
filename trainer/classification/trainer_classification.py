import abc
import numpy as np
import os

from io_data import readNifty

from ..trainer_base import TrainerBase


class TrainerClassification(TrainerBase):
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
            ids = os.path.basename(image_path).split(".")[0].split("_")
            pat_meta_data = self._get_patient_date(patient_id=ids[0], nodule_id=int(ids[-1]))

            x_ = self._get_image(image_path=image_path)
            mask = self._get_mask(mask_path=mask_path)

            X.append(x_[mask == 1])
            Y.append(np.full(shape=(np.count_nonzero(mask),), fill_value=pat_meta_data["Diagnosis_value"]))

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
