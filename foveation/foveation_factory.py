
from foveation.methods.gaze_crop import GazeCenteredCrop
from foveation.methods.radial_blur import RadialBlurFoveation
from foveation.methods.fcg import FovealCartesianGeometry


def build_foveation(fov_type):
    """
    Method to build foveation types out of given config.
    Playground for parameters.
    
    :param fov_type: name of the foveation method
    """
    
    if fov_type is None:
        return None
    elif fov_type == "none":
        return None
    elif fov_type == "crop":
        return GazeCenteredCrop(crop_size=336)
    elif fov_type == "blur":
        return RadialBlurFoveation(radii=[32, 64, 128, 256], sigmas=[0.0, 0.5, 1.5, 3.0, 6.0], saliency_alpha=1.0, transition_width=32)
    elif fov_type == "fcg":
        return FovealCartesianGeometry(p0=64, alpha=0.5, beta=1.0)
    else:
        raise ValueError(f"Unknown foveation type: {fov_type}") 