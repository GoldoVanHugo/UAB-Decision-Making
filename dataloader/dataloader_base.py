import os
import abc
import pandas as pd
import numpy as np

from io_data import readNifty
from utils import (
    shuffle,
    get_image_and_mask_paths,
    get_split_idx,
)
from constants import SIZE_3D


class DataloaderBase:
    def __init__(self, data_path: str, meta_file: str, image_folder: str = "image", mask_folder: str = "nodule_mask",
                 file_extension: str = ".nii.gz", set_seed: bool = True, use_all_voxels: bool = True, d3: bool = False):
        """
        Trainer to train an SVM model

        Arguments:

        - data_path (str): The path to the dataset with the images and the masks.
        - model_path (str): The path to where the model will be stored.
        - meta_file (str): The file with the metadata.
        - image_folder (str): The name of the folder with the images (default: image).
        - mask_folder (str): The name of the folder with the masks (default: nodule_mask).
        - file_extension (str): The file extension from the files to load (default: .nii.gz).
        - set_seed (bool): Use a seed for random operations (default: True).
        - use_all_voxels (bool): Use all voxels in training data (default: True).
        - d3 (bool): Create 3D data (default: False)
        """
        self.data_path = data_path
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.meta_file = meta_file
        self.file_extension = file_extension
        self.set_seed = set_seed
        self.use_all_voxels = use_all_voxels
        self.d3 = d3
        self.meta_data = None
        self._set_meta_data(file=self.meta_file)

    def _set_meta_data(self, file: str):
        self.meta_data = pd.read_excel(os.path.join(self.data_path, "..", file))

    def _get_patient_date(self, patient_id: str, nodule_id: int):
        pat_rows = self.meta_data.loc[self.meta_data["patient_id"] == patient_id]

        return pat_rows.loc[pat_rows["nodule_id"] == nodule_id]

    def _get_image(self, image_path: str) -> np.ndarray:
        image, _ = readNifty(filePath=image_path)

        if self.d3:
            patch_images = []
            for z in range(image.shape[-1]):
                pad_image = np.pad(
                    array=image[..., z],
                    pad_width=SIZE_3D // 2,
                    mode="reflect",
                )
                patch_image = np.lib.stride_tricks.sliding_window_view(pad_image, window_shape=(SIZE_3D, SIZE_3D))
                patch_images.append(np.expand_dims(patch_image, axis=2))

            patch_images = np.concatenate(patch_images, axis=2)

            return patch_images.reshape((-1, SIZE_3D * SIZE_3D))

        return image.reshape((-1, 1))

    @staticmethod
    def _get_mask(mask_path: str) -> np.ndarray:
        mask, _ = readNifty(filePath=mask_path)

        # flatten data
        return mask.ravel()

    @abc.abstractmethod
    def _get_dataset(self, image_paths: list[str], mask_paths: list[str], train: bool = True) -> tuple[
                     np.ndarray, np.ndarray]:
        raise ValueError("Implement in child class.")

    def get_train_and_test_datasets(self, split_ratio: float = 0.8) -> tuple[
                                    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        image_paths, mask_paths = get_image_and_mask_paths(
            data_path=self.data_path,
            image_folder=self.image_folder,
            mask_folder=self.mask_folder,
            file_extension=self.file_extension
        )
        split_idx = get_split_idx(paths=image_paths, split_ratio=split_ratio)

        train_dataset = self._get_dataset(
            image_paths=image_paths[:split_idx],
            mask_paths=mask_paths[:split_idx],
        )
        test_dataset = self._get_dataset(
            image_paths=image_paths[split_idx:],
            mask_paths=mask_paths[split_idx:],
            train=False,
        )

        return shuffle(x=train_dataset[0], y=train_dataset[1], set_seed=self.set_seed), test_dataset
