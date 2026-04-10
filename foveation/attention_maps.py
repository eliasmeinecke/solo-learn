import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from foveation.ooc.ooc_utils import load_model, load_data, IdentityFoveation
from foveation.factory import setup_exact_foveation

import torchvision.transforms.v2 as v2


OUT_DIR = Path("/home/elias/solo-learn/foveation/plots/attention_maps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_INDEX_PATH = "/home/elias/solo-learn/imagenet_class_index.json"

with open(CLASS_INDEX_PATH, "r") as f:
    CLASS_INDEX = json.load(f)

# mapping: int → imagenet class name
IDX_TO_NAME = {int(k): v[1] for k, v in CLASS_INDEX.items()}


# ------------------------
# GradCAM core
# ------------------------

def run_gradcam(model, img, label=None):

    target_layers = [model.backbone.layer4[-1]]

    with GradCAM(model=model, target_layers=target_layers) as cam:

        if label is None:
            outputs = model(img)
            pred = outputs.argmax(dim=1).item()
        else:
            pred = label

        targets = [ClassifierOutputTarget(pred)]

        grayscale_cam = cam(
            input_tensor=img,
            targets=targets,
            aug_smooth=True,
            eigen_smooth=True
        )

        cam_map = grayscale_cam[0]

    return cam_map, pred


def visualize_cam(img_tensor, cam_map):

    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

    # undo normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)

    vis = show_cam_on_image(img, cam_map, use_rgb=True)

    return vis


# ------------------------
# Main pipeline
# ------------------------

def main(dataset_name):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- post transform ---
    T_post = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # --- data ---
    loader = load_data(dataset_name)

    dataset = loader.dataset
    #indices = random.sample(range(len(dataset)), 3)
    indices = [1107, 209, 142]

    print(f"Selected indices: {indices}")

    # --- load all models ---
    models = {}
    foveations = {}

    for m in MODELS:
        model = load_model(m).to(device)
        model.eval()
        models[m] = model

        if m in ["base", "dummy"]:
            foveations[m] = IdentityFoveation().to(device)
        else:
            foveations[m] = setup_exact_foveation(m).to(device)

    # --- plot grid ---
    fig, axes = plt.subplots(len(indices), len(MODELS), figsize=(4 * len(MODELS), 4 * len(indices)))

    for row_idx, idx in enumerate(indices):

        img, label, gaze = dataset[idx]

        for col_idx, m in enumerate(MODELS):
            
            model = models[m]
            foveation = foveations[m]

            img_t = img.unsqueeze(0).to(device)
            gaze_t = gaze.unsqueeze(0).to(device)

            # --- gaze → absolute ---
            B, C, H, W = img_t.shape
            gaze_abs = gaze_t.clone()
            gaze_abs[:, 0] = gaze_t[:, 0] * W
            gaze_abs[:, 1] = gaze_t[:, 1] * H

            # --- foveation ---
            img_fov = foveation(img_t, gaze_abs)

            # --- post transform ---
            img_input = T_post(img_fov)

            # --- gradcam ---
            cam_map, pred = run_gradcam(model, img_input)
            vis = visualize_cam(img_input, cam_map)

            ax = axes[row_idx, col_idx]

            # --- labels ---
            pred_name = IDX_TO_NAME[pred].replace("_", " ").title()
            gt_name = IDX_TO_NAME[label].replace("_", " ").title()

            correct = (pred == label)

            # --- plot ---
            ax.imshow(vis)
            ax.axis("off")

            title = f"{m}\nPred: {pred_name}\nGT: {gt_name}"

            if correct:
                ax.set_title(title, color="green", fontsize=9)
            else:
                ax.set_title(title, color="red", fontsize=9)

    # --- global title ---
    plt.suptitle(f"Grad-CAM Comparison – {dataset_name}", fontsize=14)

    plt.tight_layout()

    out_path = OUT_DIR / f"comparison_{dataset_name}.png"
    plt.savefig(out_path, bbox_inches="tight")
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
    
    main(args.dataset)