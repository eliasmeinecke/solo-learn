
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from foveation.methods.radial_blur import RadialBlurFoveation
from foveation.methods.cm import CortalMagnification

class GazePredictor(nn.Module):
    """
    Produces gaze and saliency tensors for linear evaluation.

    Default:
        - gaze = image center
        - saliency = centered Gaussian blob

    Can later be extended with:
        - learned gaze predictor
        - learned saliency predictor
    """

    def __init__(self, saliency_resolution=64, gaussian_sigma_ratio=0.15):
        super().__init__()
        self.saliency_resolution = saliency_resolution
        self.gaussian_sigma_ratio = gaussian_sigma_ratio

    def forward(self, img):
        """
        img: (B, C, H, W)

        returns:
            gaze: (B, 2)
            saliency: (B, 1, H, W)
        """

        device = img.device
        B, C, H, W = img.shape

        # Center Gaze
        gaze_x = torch.full((B,), W / 2, device=device)
        gaze_y = torch.full((B,), H / 2, device=device)

        gaze = torch.stack([gaze_x, gaze_y], dim=1)  # (B, 2)

    
        # Center Gaussian Saliency
        sal_res = self.saliency_resolution

        ys = torch.arange(sal_res, device=device).view(1, sal_res, 1)
        xs = torch.arange(sal_res, device=device).view(1, 1, sal_res)

        ys = ys.expand(B, sal_res, sal_res)
        xs = xs.expand(B, sal_res, sal_res)

        cx = sal_res / 2
        cy = sal_res / 2
        sigma = self.gaussian_sigma_ratio * sal_res

        sal = torch.exp(
            -((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2)
        )

        sal = sal / (sal.amax(dim=(1, 2), keepdim=True) + 1e-6)

        sal = sal.unsqueeze(1)  # (B, 1, sal_res, sal_res)

        # Upscale saliency to image resolution
        if sal_res != H:
            sal = F.interpolate(
                sal,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        return gaze, sal
    
    
def setup_foveation(foveation_cfg):

    if foveation_cfg is None:
        return None

    fov_type = foveation_cfg.get("type", None)
    params = foveation_cfg.get(fov_type, {})

    if fov_type == "blur":
        return RadialBlurFoveation(**params)

    elif fov_type == "cm":
        return CortalMagnification(**params)

    return None


def log_foveation_config(foveation_cfg, context: str, gpu_augmentation: bool = True):
    """
    Logs foveation configuration and whether it is applied
    in the given runtime context.

    Args:
        foveation_cfg: foveation configuration dict or None
        context: "pretrain", "linear_eval", "knn"
        gpu_augmentation: whether GPU augmentation is enabled
    """

    # Only print on rank 0
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return

    print("\n" + "=" * 60)

    if foveation_cfg is None:
        print(f"[FOVEATION] ({context}) Disabled")
        print("=" * 60 + "\n")
        return

    fov_type = foveation_cfg.get("type", "none")
    params = foveation_cfg.get(fov_type, {})

    applied = False
    location = ""

    # CROP (CPU only, pretrain only)
    if fov_type == "crop":
        if context == "pretrain":
            applied = True
            location = "CPU (Dataset)"
        else:
            applied = False
            location = "Not applied in this context"

    # BLUR / CM (GPU only)
    elif fov_type in ["blur", "cm"]:
        if gpu_augmentation:
            applied = True
            location = "GPU"
        else:
            applied = False
            location = "GPU augmentation disabled"

    else:
        applied = False
        location = "Unknown type"

    print(f"[FOVEATION] ({context}) Configured: {fov_type.upper()}")

    if applied:
        print(f"Status: ACTIVE  → Applied on {location}")
    else:
        print(f"Status: INACTIVE → {location}")

    if params:
        print("Parameters:")
        for k, v in params.items():
            print(f"  - {k}: {v}")

    print("=" * 60 + "\n")
