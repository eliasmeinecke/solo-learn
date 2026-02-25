import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class RadialBlurFoveation(nn.Module):
    def __init__(
        self,
        radii_frac=[0.3, 0.7],
        sigma_base_frac=0.006,
        sigma_growth=2.0,
        saliency_alpha=5.0,
        transition_frac=0.1,
    ):
        super().__init__()

        self.radii_frac = radii_frac
        self.sigma_base_frac = sigma_base_frac
        self.sigma_growth = sigma_growth
        self.saliency_alpha = saliency_alpha
        self.transition_frac = transition_frac

    def forward(self, img, gaze, saliency):
        """
        img:       (B, C, H, W) uint8 tensor
        gaze:      (B, 2) tensor with (x, y)
        saliency:  (B, 1, H, W) float tensor in [0,1]
        """

        device = img.device
        B, C, H, W = img.shape

        img = img.float()
        
        # ensure saliency matches image size & normalize
        if saliency.shape[-2:] != (H, W):
            saliency = F.interpolate(
                saliency,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
        saliency = saliency / (saliency.amax(dim=(2,3), keepdim=True) + 1e-6)
        saliency = saliency.squeeze(1)
        
        # create coordinate grid (GPU)
        ys = torch.arange(H, device=device).view(1, H, 1)
        xs = torch.arange(W, device=device).view(1, 1, W)

        ys = ys.expand(B, H, W)
        xs = xs.expand(B, H, W)

        x_g = gaze[:, 0].view(B, 1, 1)
        y_g = gaze[:, 1].view(B, 1, 1)

        # distance map
        R = torch.sqrt((xs - x_g) ** 2 + (ys - y_g) ** 2)

        R_max = R.amax(dim=(1, 2), keepdim=True)
        R_eff = R / (1.0 + self.saliency_alpha * saliency)

        # radii & sigmas
        radii = [f * R_max for f in self.radii_frac]
        transition_width = self.transition_frac * R_max

        sigma_base = self.sigma_base_frac * min(H, W)

        sigmas = [0.0]
        for i in range(len(self.radii_frac)):
            sigmas.append(sigma_base * (self.sigma_growth ** i))

        # blurred versions
        blurred_imgs = []
        for sigma in sigmas:
            if sigma == 0:
                blurred_imgs.append(img)
            else:
                # kernel size automatically derived (maybe change logic?)
                k = int(2 * round(3 * sigma) + 1)
                blurred = TF.gaussian_blur(img, kernel_size=k, sigma=sigma)
                blurred_imgs.append(blurred)

        # ring centers
        ring_centers = []
        prev = torch.zeros_like(R_max)

        for r in radii:
            ring_centers.append(0.5 * (prev + r))
            prev = r

        ring_centers.append(prev + transition_width)

        # soft weights
        weights = []
        for c in ring_centers:
            w = torch.exp(-0.5 * ((R_eff - c) / transition_width) ** 2)
            weights.append(w)

        weights = torch.stack(weights, dim=0)
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-6)

        # weighted blending
        output = torch.zeros_like(img)

        for w, img_blur in zip(weights, blurred_imgs):
            output += w.unsqueeze(1) * img_blur

        return output.clamp(0, 255).to(torch.uint8)