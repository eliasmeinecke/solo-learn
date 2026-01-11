
import numpy as np
from PIL import Image
import pandas as pd
import cv2
from foveation.base import Foveation


"""All credits go to: https://github.com/pytorch/vision/issues/4434#issuecomment-921814114"""
class LogPolarTransform(Foveation):
    def __init__(self, max_radius=100.0):
        self.max_radius = max_radius
    
    def __call__(self, img: Image.Image, annot: pd.Series) -> Image.Image:
        assert isinstance(img, Image.Image)
    
        dsize = img.size
        x_g, y_g = annot.gaze_loc_x, annot.gaze_loc_y
        
        center = [x_g, y_g]
        
        flags = cv2.WARP_POLAR_LOG # add for inverse mapping: | cv2.WARP_INVERSE_MAP
        out = cv2.warpPolar(
            np.asarray(img), dsize=dsize, center=center, maxRadius=self.max_radius, flags=flags
        )
        return Image.fromarray(out)
    