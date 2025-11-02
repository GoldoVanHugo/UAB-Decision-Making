import abc
import os
import numpy as np
import pandas as pd

from glob import glob

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from constants import (
    SEED,
    VOXEL_TRAIN_RATIO,
)


class TrainerBase:
    def __init__(self, data_path: str, model_path: str, model_name: str, meta_file: str, image_folder: str = "image",
                 mask_folder: str = "nodule_mask", set_seed: bool = True):
        """
        Trainer to train an SVM model

        Arguments:

        - data_path (str): The path to the dataset with the images and the masks.
        - model_path (str): The path to where the model will be stored.
        - model_name (str): The name of the model to store.
        - meta_file (str): The file with the metadata.
        - image_folder (str): The name of the folder with the images (default: image).
        - mask_folder (str): The name of the folder with the masks (default: nodule_mask).
        - set_seed (boolean): Use a seed for random operations (default: True).
        """
        self.data_path = data_path
        self.model_path = model_path
        self.model_name = model_name
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.meta_file = meta_file
        self.set_seed = set_seed
        self.meta_data = None
        self._set_meta_data(file=self.meta_file)

        self.model = None

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def _set_meta_data(self, file: str):
        self.meta_data = pd.read_excel(os.path.join(self.data_path, "..", file))

    def _get_patient_date(self, patient_id: str, nodule_id: int):
        pat_rows = self.meta_data.loc[self.meta_data["patient_id"] == patient_id]

        return pat_rows.loc[pat_rows["nodule_id"] == nodule_id]

    def _get_paths(self, file_extension: str = ".nii.gz") -> tuple[list, list]:
        """
        Get all images and mask paths. The paths are sorted.

        Return

        A tuple with two list. The first list contains the image paths, the second list the mask paths.
        """
        img_paths = glob(os.path.join(self.data_path, self.image_folder, f"*{file_extension}"))

        if len(img_paths) == 0:
            raise ValueError(f"No images found in path {os.path.join(self.data_path, self.image_folder)}")

        img_paths = sorted(img_paths)
        mask_paths = []

        for img_path in img_paths:
            file_name = os.path.basename(img_path)
            mask_path = os.path.join(self.data_path, self.mask_folder, file_name)

            if not os.path.exists(mask_path) and not os.path.isfile(mask_path):
                print(f"No mask file found for image: {file_name}!")
                continue

            mask_paths.append(mask_path)

        assert len(img_paths) == len(mask_paths)

        return img_paths, mask_paths

    @staticmethod
    def _get_split(paths: list, split_ratio: float = .8) -> int:
        """
        Returns the index to split the dataset. Dataset must be sorted and all the data from one patient is in the same
        split.

        Arguments:

        - paths (list[str]): The sorted paths to split
        - split_ratio (float): The ratio to split the paths (default: 0.8)

        Returns:

        The index to split the dataset
        """

        def get_id(file: str) -> str:
            return os.path.basename(file).split(".")[0].split("_")[0]

        ids = list(set(get_id(file=file) for file in paths))
        split_ids = int(split_ratio * len(ids))

        ids_train = ids[:split_ids]
        split_idx = 0
        for p in paths:
            if get_id(file=p) in ids_train:
                split_idx += 1

        return split_idx

    @abc.abstractmethod
    def _get_image(self, image_path: str) -> np.ndarray:
        raise ValueError("Implement in child class.")

    @abc.abstractmethod
    def _get_mask(self, mask_path: str) -> np.ndarray:
        raise ValueError("Implement in child class.")

    @abc.abstractmethod
    def _get_dataset(self, image_paths: list[str], mask_paths: list[str], voxel_ratio: float = None) -> tuple[
                     np.ndarray, np.ndarray]:
        raise ValueError("Implement in child class.")

    @staticmethod
    def _scale_x(x: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()

        return scaler.fit_transform(x)

    def _shuffle(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.set_seed:
            np.random.seed(seed=SEED)

        indices = np.arange(len(x))
        np.random.shuffle(indices)

        return x[indices], y[indices]

    def get_train_and_test_datasets(self, file_extension: str = ".nii.gz", split_ratio: float = 0.8) -> tuple[
                                    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        image_paths, mask_paths = self._get_paths(file_extension=file_extension)
        split_idx = self._get_split(paths=image_paths, split_ratio=split_ratio)

        train_dataset = self._get_dataset(
            image_paths=image_paths[:split_idx],
            mask_paths=mask_paths[:split_idx],
            voxel_ratio=VOXEL_TRAIN_RATIO,
        )
        test_dataset = self._get_dataset(
            image_paths=image_paths[split_idx:],
            mask_paths=mask_paths[split_idx:],
        )

        return self._shuffle(x=train_dataset[0], y=train_dataset[1]), test_dataset

    @abc.abstractmethod
    def save_model(self):
        raise ValueError("Implement in child class.")

    @abc.abstractmethod
    def load_model(self, model_path: str):
        raise ValueError("Implement in child class.")

    @abc.abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise ValueError("Implement in child class.")

    @abc.abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        raise ValueError("Implement in child class.")

    def test(self, x_test: np.ndarray, y_test: np.ndarray):
        y_pred = self.predict(x=x_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print("=== Ergebnisse ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")

        return acc, prec, rec, f1
