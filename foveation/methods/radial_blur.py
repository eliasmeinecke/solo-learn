
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from foveation.methods.base import Foveation


class RadialBlurFoveation(Foveation):
    def __init__(self, radii=[32, 64, 128, 256], sigmas=[0.0, 0.5, 1.5, 3.0, 6.0], saliency_alpha=1.0, transition_width=32):
        
        assert len(sigmas) == len(radii) + 1, (
            f"Got {len(sigmas)} sigmas for {len(radii)+1} rings."
        )
        
        self.radii = radii
        self.sigmas = sigmas
        self.saliency_alpha = saliency_alpha
        self.transition_width = transition_width

    def __call__(self, img: Image.Image, annot: pd.Series, saliency: np.ndarray) -> Image.Image:
        
        img_np = np.array(img)
        H, W, _ = img_np.shape

        x_g, y_g = annot.gaze_loc_x, annot.gaze_loc_y
        
        S = cv2.resize(saliency, (W, H), interpolation=cv2.INTER_LINEAR)
        # normalize saliency just to be safe
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

        ring_centers = []
        prev = 0.0
        for r in self.radii:
            ring_centers.append(0.5 * (prev + r))
            prev = r
        ring_centers.append(prev + self.transition_width)
        
        weights = []
        for c in ring_centers:
            w = np.exp(-0.5 * ((R_eff - c) / self.transition_width) ** 2)
            weights.append(w)
        weights = np.stack(weights, axis=0)
        weights /= np.sum(weights, axis=0, keepdims=True) + 1e-6
        
        output = np.zeros_like(img_np, dtype=np.float32)

        for w, img_blur in zip(weights, blurred_imgs):
            output += w[..., None] * img_blur

        return Image.fromarray(output.astype(np.uint8))
    