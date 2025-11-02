import os.path

import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier

from io_data import readNifty
from .trainer_segmentation import TrainerSegmentation
from constants import SIZE_3D


class TrainerRandomForest(TrainerSegmentation):
    def _get_image(self, image_path: str) -> np.ndarray:
        image, _ = readNifty(filePath=image_path)

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

        return patch_images.reshape((-1, SIZE_3D*SIZE_3D))

    def save_model(self):
        if self.model is None:
            raise ValueError("First train or load a model.")

        joblib.dump(self.model, os.path.join(self.model_path, f"{self.model_name}.joblib"))

    def load_model(self, model_path: str):
        self.model = joblib.load(model_path)

    def train(self, x: np.ndarray, y: np.ndarray):
        if self.model is None:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                n_jobs=-1,  # use all CPUS
                random_state=42,
                verbose=2
            )

        self.model.fit(x, y)

    def predict(self, x:np.ndarray):
        if self.model is None:
            raise ValueError("First train or load a model.")

        return self.model.predict(x)
