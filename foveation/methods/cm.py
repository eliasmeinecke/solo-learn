import torch
import torch.nn as nn
import torch.nn.functional as F


class CortalMagnification(nn.Module):
    def __init__(self, fov_ratio=0.3125, K_ratio=0.208, saliency_beta=0.5):
        super().__init__()
        self.fov_ratio = fov_ratio
        self.K_ratio = K_ratio
        self.saliency_beta = saliency_beta

    def forward(self, img, gaze, saliency):
        """
        img:       (B, C, H, W) uint8
        gaze:      (B, 2)
        saliency:  (B, 1, H, W)
        """

        device = img.device
        B, C, H, W = img.shape

        img = img.float()
        
        # ensure saliency matches image size & normalize
        if saliency.shape[-2:] != (H, W):
            saliency = torch.nn.functional.interpolate(
                saliency,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
        saliency = saliency / (saliency.amax(dim=(2,3), keepdim=True) + 1e-6)
        saliency = saliency.squeeze(1)

        # coordinate grid (batchfähig)
        ys = torch.arange(H, device=device).view(1, H, 1)
        xs = torch.arange(W, device=device).view(1, 1, W)

        ys = ys.expand(B, H, W)
        xs = xs.expand(B, H, W)

        x_g = gaze[:, 0].view(B, 1, 1)
        y_g = gaze[:, 1].view(B, 1, 1)

        dx = xs - x_g
        dy = ys - y_g

        r = torch.sqrt(dx**2 + dy**2 + 1e-6)

        mean_r = torch.sum(saliency * r, dim=(1, 2), keepdim=True)
        mean_r2 = torch.sum(saliency * r**2, dim=(1, 2), keepdim=True)

        var_r = mean_r2 - mean_r**2
        std_r = torch.sqrt(torch.clamp(var_r, min=0.0))

        spread_norm = std_r / r.amax(dim=(1, 2), keepdim=True)

        # parameters
        base_size = min(H, W)

        fov = self.fov_ratio * base_size
        K = self.K_ratio * base_size

        fov_eff = fov * (1.0 + self.saliency_beta * spread_norm)

        # radial transform
        r_new = radial_quadratic_batch(r, fov_eff, K)

        dx_norm = dx / (r + 1e-6)
        dy_norm = dy / (r + 1e-6)

        X_new = x_g + dx_norm * r_new
        Y_new = y_g + dy_norm * r_new
        
        # normalize for grid_sample
        X_norm = (X_new / (W - 1)) * 2 - 1
        Y_norm = (Y_new / (H - 1)) * 2 - 1

        grid = torch.stack((X_norm, Y_norm), dim=-1)

        out = F.grid_sample(
            img,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        return out.clamp(0, 255).to(torch.uint8)


def radial_quadratic_batch(r, fov, K):

    r_out = (r + K)**2 / (2 * (fov + K)) + (fov - K) / 2

    r_tfm = torch.where(r < fov, r, r_out)

    coef = r.amax(dim=(1,2), keepdim=True) / (
        r_tfm.amax(dim=(1,2), keepdim=True) + 1e-6
    )

    return coef * r_tfm