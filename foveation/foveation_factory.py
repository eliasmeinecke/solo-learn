
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
        return GazeCenteredCrop()
    elif fov_type == "blur":
        return RadialBlurFoveation()
    elif fov_type == "fcg":
        return FovealCartesianGeometry()
    else:
        raise ValueError(f"Unknown foveation type: {fov_type}") 