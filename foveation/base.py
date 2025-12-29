
from abc import ABC, abstractmethod
from PIL import Image
import pandas as pd


class Foveation(ABC):
    @abstractmethod
    def __call__(self, img: Image.Image, annot: pd.Series) -> Image.Image:
        pass