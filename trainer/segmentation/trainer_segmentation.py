import abc
import numpy as np

from io_data import readNifty

from ..trainer_base import TrainerBase


class TrainerSegmentation(TrainerBase):
    @abc.abstractmethod
    def _get_image(self, image_path: str) -> np.ndarray:
        return super()._get_image(image_path=image_path)

    def _get_mask(self, mask_path: str) -> np.ndarray:
        mask, _ = readNifty(filePath=mask_path)

        # flatten data
        return mask.ravel()

    @abc.abstractmethod
    def save_model(self):
        super().save_model()

    @abc.abstractmethod
    def load_model(self, model_path: str):
        super().load_model(model_path=model_path)

    @abc.abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray):
        super().train(x=x, y=y)

    @abc.abstractmethod
    def predict(self, x:np.ndarray) -> np.ndarray:
        return super().predict(x=x)
    