
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import pandas as pd


class Foveation(ABC):
    @abstractmethod
    def __call__(self, img: Image.Image, annot: pd.Series, saliency: np.ndarray) -> Image.Image:
        pass