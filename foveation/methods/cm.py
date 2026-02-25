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
        
        # saliency-weighted spread
        S = saliency.squeeze(1)
        S = S / (S.sum(dim=(1, 2), keepdim=True) + 1e-6)

        mean_r = torch.sum(S * r, dim=(1, 2), keepdim=True)
        mean_r2 = torch.sum(S * r**2, dim=(1, 2), keepdim=True)

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

        dx_norm = dx / r
        dy_norm = dy / r

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

    r_tfm = torch.zeros_like(r)

    mask_fovea = r < fov

    # inside fovea
    r_tfm[mask_fovea] = r[mask_fovea]

    # outside fovea
    r_out = r[~mask_fovea]

    r_tfm[~mask_fovea] = (
        (r_out + K)**2 / (2 * (fov + K))
        + (fov - K) / 2
    )

    coef = r.amax(dim=(1, 2), keepdim=True) / (
        r_tfm.amax(dim=(1, 2), keepdim=True) + 1e-6
    )

    r_new = coef * r_tfm

    return r_new