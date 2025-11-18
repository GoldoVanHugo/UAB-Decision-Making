import os

import numpy as np
import joblib

from sklearn.svm import LinearSVC

from .trainer_base import TrainerBase


class TrainerSVM(TrainerBase):
    def save_model(self):
        if self.model is None:
            raise ValueError("First train or load a model.")

        joblib.dump(self.model, os.path.join(self.model_path, f"{self.model_name}.joblib"))

    def load_model(self, model_path: str):
        self.model = joblib.load(model_path)

    def train(self, x: np.ndarray, y: np.ndarray):
        if self.model is None:
            self.model = LinearSVC(
                verbose=1,
            )

        self.model.fit(x, y)

    def predict(self, x: np.ndarray):
        if self.model is None:
            raise ValueError("First train or load a model.")

        return self.model.predict(x)
