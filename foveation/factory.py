
import numpy as np

from foveation.methods.radial_blur import RadialBlurFoveation
from foveation.methods.cm import CortalMagnification

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
    
    
def setup_foveation(foveation_cfg):
    
    if foveation_cfg is None:
        return None

    fov_type = foveation_cfg.get("type", None)
    params = foveation_cfg.get(fov_type, {})
    if fov_type == "blur":
        foveation = RadialBlurFoveation(**params)
    elif fov_type == "cm":
        foveation = CortalMagnification(**params)
    else:
        foveation = None
        
    return foveation
