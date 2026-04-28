import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from foveation.utils import build_model_and_foveation, T_POST, load_imagenet_class_map
from foveation.ooc.ooc_utils import load_data
from foveation.FovEx import FovExWrapper


OUT_DIR = Path("/home/elias/solo-learn/foveation/plots/attention_maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IDX_TO_NAME = load_imagenet_class_map()


# FovEx
def build_fovex(model, device):
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    def target_function(x, y):
        return y
    fovex = FovExWrapper(
        downstream_model=model,
        criterion=criterion,
        target_function=target_function,
        image_size=224,
        foveation_sigma=0.15,
        blur_filter_size=41,
        blur_sigma=10,
        forgetting=0.1,
        foveation_aggregation=1,
        heatmap_sigma=0.15,
        heatmap_forgetting=[1.0]*10,
        device=device
    )
    return fovex


def visualize_fovex(img_tensor, heatmap):
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    cmap = plt.cm.jet(heatmap)[..., :3]
    vis = 0.5 * img + 0.5 * cmap
    vis = np.clip(vis, 0, 1)
    return vis


def run_fovex(fovex, img, target_class, device):
    # img must be [0,1] (NOT normalized!)
    # prediction target
    target = torch.tensor([target_class]).to(device)
    explanation, fixations, _, _ = fovex.generate_explanation(
        img,
        target,
        scanpath_length=10,
        opt_iterations=20,
        learning_rate=0.1,
        random_restarts=True,
        normalize_heatmap=True
    )
    heatmap = explanation[0, 0].detach().cpu().numpy()
    fixations = fixations.detach().cpu().numpy()
    return heatmap, fixations


# GradCAM
def run_gradcam(model, img, target_class):
    target_layers = [model.backbone.layer4[-1]]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(
            input_tensor=img,
            targets=targets,
            aug_smooth=True,
            eigen_smooth=True
        )
        cam_map = grayscale_cam[0]
    return cam_map


def visualize_cam(img_tensor, cam_map):
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    # undo normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    vis = show_cam_on_image(img, cam_map, use_rgb=True)
    return vis


# Main pipeline
def main(dataset_name, indices):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- data ---
    loader = load_data(dataset_name)
    dataset = loader.dataset

    # --- models ---
    models = {}
    foveations = {}
    fovex_models = {}

    for m in MODELS:
        model, foveation = build_model_and_foveation(m, device)
        models[m] = model
        foveations[m] = foveation
        fovex_models[m] = build_fovex(model, device)
        
    for idx in indices:
        img, label, gaze = dataset[idx]

        # --- figure ---
        fig, axes = plt.subplots(
            2, len(MODELS),
            figsize=(3.5 * len(MODELS), 5),
            gridspec_kw={"wspace": 0.05, "hspace": 0.1}
        )

        # --- GT label ---
        gt_name = IDX_TO_NAME[label]

        for col_idx, m in enumerate(MODELS):

            model = models[m]
            foveation = foveations[m]
            fovex = fovex_models[m]

            img_t = img.unsqueeze(0).to(device)
            gaze_t = gaze.unsqueeze(0).to(device)

            # --- gaze → absolute ---
            B, C, H, W = img_t.shape
            gaze_abs = gaze_t.clone()
            gaze_abs[:, 0] *= W
            gaze_abs[:, 1] *= H

            # --- foveation ---
            img_fov = foveation(img_t, gaze_abs)

            # --- FovEx input ---
            img_fov_224 = img_fov.float() / 255.0
            img_fov_224 = F.interpolate(
                img_fov_224,
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            )

            # --- visualization base ---
            img_fov_vis = img_fov.float() / 255.0

            # --- model input ---
            img_input = T_POST(img_fov)

            # --- prediction ---
            outputs = model(img_input)
            pred = outputs.argmax(dim=1).item()
            pred_name = IDX_TO_NAME[pred]

            correct = (pred == label)
            color = "green" if correct else "red"

            # ======================
            # GradCAM
            # ======================
            cam_map = run_gradcam(model, img_input, label)
            vis_cam = visualize_cam(img_input, cam_map)

            ax = axes[0, col_idx]
            ax.imshow(vis_cam)
            ax.axis("off")

            ax.set_title(
                f"{m}\n{pred_name}",
                color=color,
                fontsize=10
            )

            # ======================
            # FovEx
            # ======================
            heatmap, fixations = run_fovex(fovex, img_fov_224, label, device)

            heatmap_resized = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
            heatmap_resized = F.interpolate(
                heatmap_resized,
                size=img_fov_vis.shape[-2:],
                mode="bilinear",
                align_corners=False
            ).squeeze().cpu().numpy()

            H, W = img_fov_vis.shape[-2:]
            y_fix = (fixations[:, 0] + 1) * (H / 2)
            x_fix = (fixations[:, 1] + 1) * (W / 2)

            vis_fovex = visualize_fovex(img_fov_vis, heatmap_resized)

            ax = axes[1, col_idx]
            ax.imshow(vis_fovex)
            #ax.plot(x_fix, y_fix, 'x', color='yellow', markersize=4)
            ax.axis("off")

        plt.tight_layout(rect=[0.08, 0, 1, 0.95])
        
        left = min(ax.get_position().x0 for ax in axes.flatten())
        x_text = left - 0.04

        y0 = axes[0, 0].get_position().y0 + axes[0, 0].get_position().height / 2
        y1 = axes[1, 0].get_position().y0 + axes[1, 0].get_position().height / 2

        fig.text(x_text, y0, "Grad-CAM", rotation=90, va="center", fontsize=12)
        fig.text(x_text, y1, "FovEx", rotation=90, va="center", fontsize=12)
        
        # --- global title ---
        plt.suptitle(
            f"Ground Truth: {gt_name}",
            fontsize=14,
            y=1
        )

        out_path = OUT_DIR / f"comparison_{idx}_{dataset_name}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Saved → {out_path}")


# ------------------------
# CLI
# ------------------------

if __name__ == "__main__":

    MODELS = ["base", "crop", "cm-strong"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="original")
    args = parser.parse_args()
    
    # look at: 501, 42,  
    # include original: 84, 42, 1107, 142?, 228, 
    # include inpainted:
    # include object: 228
    # include ooc: 84, 42, 142, 228, 
    # tested: 69, 302, 209, 51, 563
    indices = [1]
    main(args.dataset, indices)
    
    