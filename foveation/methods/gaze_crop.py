
from PIL import Image
import pandas as pd
import numpy as np
from foveation.methods.base import Foveation


class GazeCenteredCrop(Foveation):
    def __init__(self, crop_size=336):
        self.N = crop_size

    def __call__(self, img: Image.Image, annot: pd.Series, saliency: np.ndarray) -> Image.Image:
        w, h = img.size
        x_g, y_g = annot.gaze_loc_x, annot.gaze_loc_y

        x_cor = x_g - max(0, x_g + self.N//2 - w) - min(0, x_g - self.N//2)
        y_cor = y_g - max(0, y_g + self.N//2 - h) - min(0, y_g - self.N//2)
        
        return img.crop((x_cor - self.N//2, y_cor - self.N//2, x_cor + self.N//2, y_cor + self.N//2))