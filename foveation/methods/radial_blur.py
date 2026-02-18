
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from foveation.methods.base import Foveation


class RadialBlurFoveation(Foveation):
    def __init__(self, radii_frac=[0.1, 0.2, 0.4, 0.8], sigma_base_frac=0.0015, sigma_growth=2, saliency_alpha=1.0, transition_frac=0.1):
        # sigma_base_frac was chosen to result in a base_sigma of ~0.8 for an image size of 540x540
        
        self.radii_frac = radii_frac
        self.sigma_base_frac = sigma_base_frac
        self.sigma_growth = sigma_growth
        self.saliency_alpha = saliency_alpha
        self.transition_frac = transition_frac

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
        R_max = np.max(R)
        R_eff = R / (1.0 + self.saliency_alpha * S)
        
        radii = [f * R_max for f in self.radii_frac]
        transition_width = self.transition_frac * R_max
        sigma_base = self.sigma_base_frac * min(H, W)
        sigmas = [
            sigma_base * (self.sigma_growth ** i)
            for i in range(len(self.radii_frac) + 1)
        ]

        blurred_imgs = []
        for sigma in sigmas:
            if sigma == 0:
                blurred_imgs.append(img_np)
            else:
                blurred_imgs.append(cv2.GaussianBlur(img_np, (0, 0), sigma))  # let opencv calculate kernel size

        ring_centers = []
        prev = 0.0
        for r in radii:
            ring_centers.append(0.5 * (prev + r))
            prev = r
        ring_centers.append(prev + transition_width)
        
        weights = []
        for c in ring_centers:
            w = np.exp(-0.5 * ((R_eff - c) / transition_width) ** 2)
            weights.append(w)
        weights = np.stack(weights, axis=0)
        weights /= np.sum(weights, axis=0, keepdims=True) + 1e-6
        
        output = np.zeros_like(img_np, dtype=np.float32)

        for w, img_blur in zip(weights, blurred_imgs):
            output += w[..., None] * img_blur

        return Image.fromarray(output.astype(np.uint8))
    