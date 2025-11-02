import os.path

import numpy as np
import joblib

from sklearn import svm

from io_data import readNifty
from .trainer_segmentation import TrainerSegmentation


class TrainerSVM(TrainerSegmentation):
    def _get_image(self, image_path: str) -> np.ndarray:
        image, _ = readNifty(filePath=image_path)

        return image.reshape((-1, 1))

    def save_model(self):
        if self.model is None:
            raise ValueError("First train or load a model.")

        joblib.dump(self.model, os.path.join(self.model_path, f"{self.model_name}.joblib"))

    def load_model(self, model_path: str):
        self.model = joblib.load(model_path)

    def train(self, x: np.ndarray, y: np.ndarray):
        if self.model is None:
            self.model = svm.SVC(
                kernel="linear",
                verbose=True,
            )

        self.model.fit(x, y)

    def predict(self, x: np.ndarray):
        if self.model is None:
            raise ValueError("First train or load a model.")

        return self.model.predict(x)
