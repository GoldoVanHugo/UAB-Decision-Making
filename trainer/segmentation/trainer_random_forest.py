import os.path

import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier

from io_data import readNifty
from .trainer_segmentation import TrainerSegmentation


class TrainerRandomForest(TrainerSegmentation):
    def _get_image(self, image_path: str) -> np.ndarray:
        image, _ = readNifty(filePath=image_path)
        # Extract coordinates grid
        z_idx, y_idx, x_idx = np.meshgrid(
            np.arange(image.shape[2]),
            np.arange(image.shape[1]),
            np.arange(image.shape[0]),
            indexing="ij"
        )

        # flatten features
        intensity = image.ravel()
        x_feat = x_idx.ravel()
        y_feat = y_idx.ravel()
        z_feat = z_idx.ravel()

        # Feature-Matrix für dieses Volume
        return np.stack([intensity, x_feat, y_feat, z_feat], axis=1)

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
