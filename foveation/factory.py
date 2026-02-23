from typing import Optional

import numpy as np
import cv2

from foveation.methods.gaze_crop import GazeCenteredCrop
from foveation.methods.radial_blur import RadialBlurFoveation
from foveation.methods.fcg import FovealCartesianGeometry


class CenterGaze:
    def __init__(self, x, y):
        self.gaze_loc_x = x
        self.gaze_loc_y = y
        

class FoveationTransform:
    def __init__(self, foveation, base_transform):
        self.foveation = foveation
        self.base_transform = base_transform

    def _build_center_annotation(self, img):
        W, H = img.size
        return CenterGaze(W // 2, H // 2)

    def _build_center_saliency(self):
        sal_res = 64
        xs = np.arange(sal_res)
        ys = np.arange(sal_res)
        X, Y = np.meshgrid(xs, ys)

        cx = sal_res // 2
        cy = sal_res // 2
        sigma = 0.15 * sal_res

        S = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        S = (S - S.min()) / (S.max() - S.min() + 1e-6)

        return S.astype(np.float32)

    def __call__(self, img):

        if self.foveation is not None:
            # later add gaze/saliency-predictors?
            annot = self._build_center_annotation(img)
            saliency = self._build_center_saliency()
            img = self.foveation(img, annot, saliency)
            
        return self.base_transform(img)
    
    
def build_foveation(fov_cfg: Optional[dict]):
    if fov_cfg is None:
        print("[Foveation] None")
        return None

    fov_type = fov_cfg.get("type", None)
    if fov_type not in ["crop", "blur", "fcg"]: # adjust later if adding another type
        print("[Foveation] Disabled")
        return None

    params = fov_cfg.get(fov_type, {})

    print(
        f"[Foveation] type={fov_type} | "
        + ", ".join(f"{k}={v}" for k, v in params.items())
    )

    if fov_type == "crop":
        return GazeCenteredCrop(**params)
    if fov_type == "blur":
        return RadialBlurFoveation(**params)
    if fov_type == "fcg":
        return FovealCartesianGeometry(**params)

    raise ValueError(f"Unknown foveation type: {fov_type}")