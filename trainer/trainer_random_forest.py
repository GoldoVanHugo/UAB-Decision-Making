import os

import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier

from .trainer_base import TrainerBase


class TrainerRandomForest(TrainerBase):
    def model_extension(self):
        return ".joblib"

    def save_model_(self):
        if self.model is None:
            raise ValueError("First train or load a model.")

        joblib.dump(self.model, os.path.join(self.model_path, f"rf_{self.model_name}{self.model_extension()}"))

    def load_model(self, model_path: str):
        self.model = joblib.load(model_path)

    def train(self, x: np.ndarray, y: np.ndarray):
        if self.model is None:
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=15,
                n_jobs=-1,  # use all CPUS
                random_state=42,
                verbose=1
            )

        self.model.fit(x, y)

    def predict(self, x:np.ndarray):
        if self.model is None:
            raise ValueError("First train or load a model.")

        scores = self.model.predict_proba(x)[:, 1]
        y_pred = (scores >= 0.5).astype(int)

        return scores, y_pred
