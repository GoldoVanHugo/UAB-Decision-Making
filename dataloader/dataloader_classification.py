import os
import numpy as np

from .dataloader_base import DataloaderBase
from utils import standard_scaling


class DataloaderClassification(DataloaderBase):
    def _get_dataset(self, image_paths: list[str], mask_paths: list[str], train: bool = True) -> tuple[
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

        return X, Y
