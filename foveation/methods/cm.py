
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from foveation.methods.base import Foveation


"""
Implementation of: "On the use of Cortical Magnification and Saccades as Biological Proxies for Data Augmentation."
Authors: Binxu Wang, David Mayo3, Arturo Deza, Andrei Barbu, Colin Conwell
"""
class CortalMagnification(Foveation):
    def __init__(self, fov_ratio=0.3125, K_ratio=0.208, saliency_beta=0.5):
        self.fov_ratio = fov_ratio
        self.K_ratio = K_ratio
        self.saliency_beta = saliency_beta

    def __call__(self, img: Image.Image, annot: pd.Series, saliency: np.ndarray) -> Image.Image:

        img_np = np.array(img)
        H, W, C = img_np.shape
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        img_t = torch.from_numpy(img_np).float().permute(2, 0, 1)  # C,H,W
        img_t = img_t.unsqueeze(0).to(device)  # 1,C,H,W

        x_g = float(annot.gaze_loc_x)
        y_g = float(annot.gaze_loc_y)
        
        S = cv2.resize(saliency, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        S = torch.from_numpy(S).to(device)
        S = S / (S.sum() + 1e-6) # convert saliency to "probability distribution"

        ys = torch.linspace(0, H - 1, H, device=device)
        xs = torch.linspace(0, W - 1, W, device=device)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")

        dx = X - x_g
        dy = Y - y_g

        r = torch.sqrt(dx**2 + dy**2 + 1e-6)
        
        mean_r = torch.sum(S * r)
        mean_r2 = torch.sum(S * r**2)
        var_r = mean_r2 - mean_r**2
        std_r = torch.sqrt(torch.clamp(var_r, min=0.0))

        spread_norm = std_r / r.max()
        
        base_size = min(H, W)
        fov = self.fov_ratio * base_size
        K = self.K_ratio * base_size

        fov_eff = fov * (1.0 + self.saliency_beta * spread_norm)
        
        # radial transform
        r_new = radial_quadratic(r, fov_eff, K)
        
        dx_norm = dx / r
        dy_norm = dy / r

        X_new = x_g + dx_norm * r_new
        Y_new = y_g + dy_norm * r_new

        # normalize for grid_sample
        X_norm = (X_new / (W - 1)) * 2 - 1
        Y_norm = (Y_new / (H - 1)) * 2 - 1

        grid = torch.stack((X_norm, Y_norm), dim=-1)
        grid = grid.unsqueeze(0)  # 1,H,W,2

        out = F.grid_sample(
            img_t,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True, # to get rid of warning
        )

        out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = np.clip(out, 0, 255).astype(np.uint8)

        return Image.fromarray(out)
    
    
def radial_quadratic(r, fov, K):
    
    mask_fovea = r < fov
    r_tfm = torch.zeros_like(r)

    # inside fovea
    r_tfm[mask_fovea] = r[mask_fovea]

    # outside fovea
    r_out = r[~mask_fovea]
    r_tfm[~mask_fovea] = (
        (r_out + K)**2 / (2 * (fov + K))
        + (fov - K) / 2
    )

    # global normalization
    coef = r.max() / r_tfm.max()
    r_new = coef * r_tfm
    
    return r_new