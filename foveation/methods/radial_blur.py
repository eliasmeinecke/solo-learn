
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from foveation.methods.base import Foveation


class RadialBlurFoveation(Foveation):
    def __init__(self, radii=[32, 64, 128, 256], sigmas=[0.0, 0.5, 1.5, 3.0, 6.0], saliency_alpha=1.0):
        
        assert len(sigmas) == len(radii) + 1, (
            f"Got {len(sigmas)} sigmas for {len(radii)+1} rings."
        )
        
        self.radii = radii
        self.sigmas = sigmas
        self.saliency_alpha = saliency_alpha

    def __call__(self, img: Image.Image, annot: pd.Series, saliency: np.ndarray) -> Image.Image:
        
        img_np = np.array(img)
        H, W, _ = img_np.shape

        x_g, y_g = annot.gaze_loc_x, annot.gaze_loc_y
        
        S = cv2.resize(saliency, (W, H), interpolation=cv2.INTER_LINEAR)
        S = S.astype(np.float32)
        S = (S - S.min()) / (S.max() - S.min() + 1e-6)

        ys = np.arange(H)
        xs = np.arange(W)
        X, Y = np.meshgrid(xs, ys)

        R = np.sqrt((X - x_g)**2 + (Y - y_g)**2)
        R_eff = R / (1.0 + self.saliency_alpha * S)

        blurred_imgs = []
        for sigma in self.sigmas:
            if sigma == 0:
                blurred_imgs.append(img_np)
            else:
                blurred_imgs.append(cv2.GaussianBlur(img_np, (0, 0), sigma))  # let opencv calculate kernel size

        output = np.zeros_like(img_np, dtype=np.float32)

        for i in range(len(self.sigmas)):
            r_min = 0 if i == 0 else self.radii[i-1]
            r_max = self.radii[i] if i < len(self.radii) else np.inf

            mask = (R_eff >= r_min) & (R_eff < r_max)
            output += mask[..., None] * blurred_imgs[i]

        return Image.fromarray(output.astype(np.uint8))
