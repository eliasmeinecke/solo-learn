import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
import numpy as np


class GazeCenteredCrop():
    def __init__(self, crop_size=336):
        self.crop_ratio = crop_size / 540

    def __call__(self, img: Image.Image, annot: pd.Series) -> Image.Image:
        w, h = img.size
        x_g, y_g = annot.gaze_loc_x, annot.gaze_loc_y
        
        N = int(self.crop_ratio * max(w, h))
        half = N // 2

        x_cor = x_g - max(0, x_g + half - w) - min(0, x_g - half)
        y_cor = y_g - max(0, y_g + half - h) - min(0, y_g - half)
        
        return img.crop((x_cor - half, y_cor - half, x_cor + half, y_cor + half))
    
    
class GazeCenteredCropGPU(nn.Module):
    def __init__(self, crop_size=336):
        super().__init__()
        self.crop_ratio = crop_size / 540

    def forward(self, img, gaze, saliency=None):
        """
        img:  (B, C, H, W) uint8
        gaze: (B, 2) absolute coordinates
        """

        B, C, H, W = img.shape
        assert B == 1, "Crop currently expects batch size 1"

        # gaze coordinates
        x_g = gaze[0, 0]
        y_g = gaze[0, 1]

        # crop size
        N = int(self.crop_ratio * max(H, W))
        half = N // 2

        # clamp center so crop stays inside image
        x_cor = torch.clamp(x_g, min=half, max=W - half)
        y_cor = torch.clamp(y_g, min=half, max=H - half)

        # crop bounds
        x1 = int(x_cor - half)
        x2 = int(x_cor + half)

        y1 = int(y_cor - half)
        y2 = int(y_cor + half)

        # tensor crop
        img = img[:, :, y1:y2, x1:x2]

        return img