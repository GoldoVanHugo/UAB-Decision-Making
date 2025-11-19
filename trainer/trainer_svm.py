import os

import numpy as np
import joblib

from sklearn.svm import LinearSVC, SVC

from .trainer_base import TrainerBase


class TrainerSVM(TrainerBase):
    def model_extension(self):
        return ".joblib"

    def save_model_(self):
        if self.model is None:
            raise ValueError("First train or load a model.")

        joblib.dump(self.model, os.path.join(self.model_path, f"svm_{self.model_name}{self.model_extension()}"))

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

        scores = self.model.decision_function(x)
        y_pred = (scores > 0).astype(int)

        return scores, y_pred
