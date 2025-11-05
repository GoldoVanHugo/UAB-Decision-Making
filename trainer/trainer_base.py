import abc
import os
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TrainerBase:
    def __init__(self, model_path: str, model_name: str, set_seed: bool = True):
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
        self.model_path = model_path
        self.model_name = model_name
        self.set_seed = set_seed

        self.model = None

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

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
