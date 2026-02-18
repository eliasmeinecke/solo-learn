
from foveation.methods.gaze_crop import GazeCenteredCrop
from foveation.methods.radial_blur import RadialBlurFoveation
from foveation.methods.fcg import FovealCartesianGeometry


class FoveationTransform:
    def __init__(self, foveation, base_transform):
        self.foveation = foveation
        self.base_transform = base_transform

    def __call__(self, img):
        # later maybe add gaze-prediction here?
        if self.foveation is not None:
            img = self.foveation(img, None)  # center fixation
        return self.base_transform(img)
    
    
def build_foveation(fov_cfg: dict | None):
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