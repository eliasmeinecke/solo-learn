
import io
import numpy as np
import pandas as pd
import random
import h5py
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pycocotools import mask as mask_utils

import torch
from torchvision.transforms import PILToTensor
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import pil_to_tensor
from torchvision.datasets import ImageFolder

from foveation.factory import setup_exact_foveation

from foveation.mask_centroids import compute_centroid, fill_mask_holes_floodfill
from foveation.ooc.ooc_data import OOCOriginalDataset, OOCInpaintedDataset
from foveation.ooc.full_imagenet_ooc import ImageNetValOOC
from foveation.crowding.crowding_helper import CrowdingDataset
from foveation.utils import build_filename_to_label_map, load_imagenet_class_map, IMAGENET_VAL_PATH, MASK_JSON_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    indices = [42_000] # 100_003, 299_507, 180_000, 676_000, 42_000, 968_200, 25_000

    samples = []

    for i in indices:

        frame, annot, saliency = load_sample(i)

        img_tensor, gaze_tensor, sal_tensor = prepare_tensors(
            frame, annot, saliency
        )

        samples.append({
            "frame": frame,
            "img_tensor": img_tensor,
            "gaze_tensor": gaze_tensor,
            "sal_tensor": sal_tensor,
            "annot": annot
        })
    
    #viz_ego4d_example(frame, annot, saliency)
    #viz_fov(samples, method="crop")
    #viz_fov(samples, method="blur")
    #viz_fov(samples, method="cm")
    #viz_blur_heatmaps(samples)
    #viz_mask_centroids()
    #viz_imagenet_mask_samples(4)
    #viz_imagenet_fov_samples()
    #viz_ooc_background_examples()
    #viz_full_imagenet_ooc()
    #viz_crowding_overview()
    
    
FNAME_TO_IDX = build_filename_to_label_map()    
IDX_TO_LABEL = load_imagenet_class_map()


def viz_fov(samples, method="cm"):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if method == "crop":
        methods = ["crop"]
    elif method == "blur":
        methods = ["blur-light", "blur-nosal", "blur-strong"]
    elif method == "cm":
        methods = ["cm-light", "cm-nosal", "cm-strong"]
    else:
        raise ValueError(method)

    # nicer names
    name_map = {
        "blur-light": "Blur-Light",
        "blur-nosal": "Blur-Medium",
        "blur-strong": "Blur-Strong",
        "cm-light": "CM-Light",
        "cm-nosal": "CM-Medium",
        "cm-strong": "CM-Strong",
        "crop": "Gaze-Crop"
    }

    n = len(samples)

    n_cols = 1 + len(methods)  # original + variants

    fig, axes = plt.subplots(n, n_cols, figsize=(4*n_cols, 4*n))

    if n == 1:
        axes = axes[None, :]  # make 2D

    # --- setup foveations ---
    fovs = {
        m: setup_exact_foveation(m).to(device)
        for m in methods
    }

    for i, sample in enumerate(samples):

        frame = sample["frame"]
        img_tensor = sample["img_tensor"].to(device)
        gaze_tensor = sample["gaze_tensor"].to(device)
        annot = sample["annot"]

        # --- original ---
        axes[i, 0].imshow(frame)
        axes[i, 0].scatter(
            annot.gaze_loc_x,
            annot.gaze_loc_y,
            c="red",
            s=20
        )
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        # --- foveations ---
        for j, m in enumerate(methods, start=1):

            fov = fovs[m]

            with torch.no_grad():
                out_tensor = fov(img_tensor, gaze_tensor)

            out = (
                out_tensor.squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            
            axes[i, j].imshow(out)
        
            if method == "crop":
                h, w = out.shape[:2]
                gx, gy = w / 2, h / 2
            else:
                gx, gy = annot.gaze_loc_x, annot.gaze_loc_y
            
            axes[i, j].scatter(gx, gy, c="red", s=20)
                

            axes[i, j].set_title(name_map[m])
            axes[i, j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    file_name = f"ego4d_{method}_comparison.png"
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / method / file_name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {file_name}")


def viz_ego4d_example(frame, annot, saliency):
    flat_max = saliency.argmax()
    max_x, max_y = flat_max % 64, flat_max // 64
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.array(frame))
    ax[1].imshow(saliency)
    ax[0].axis("off")
    ax[1].axis("off")
    ax[0].scatter(annot.gaze_loc_x, annot.gaze_loc_y, c="red", s=50)
    ax[1].scatter(max_x, max_y, c="red", s=50)
    plt.tight_layout()
    
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / "ego4d_saliency" / "ego4d_example.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved ego4d_example.png")  


def viz_blur_heatmaps(samples):
    radii_frac = [0.3, 0.7]
    sigma_base_frac = 0.006
    sigma_growth = 2
    transition_frac = 0.1
    
    sample = samples[0]
    img = sample["frame"]
    img_tensor = sample["img_tensor"].to(device)
    gaze_tensor = sample["gaze_tensor"].to(device)
    
    _, _, H, W = img_tensor.shape

    x_g = gaze_tensor[:, 0]
    y_g = gaze_tensor[:, 1]
    
    # --- coordinate grid ---
    ys = torch.arange(H, device=img_tensor.device)
    xs = torch.arange(W, device=img_tensor.device)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")

    x_g = x_g.view(-1, 1, 1)
    y_g = y_g.view(-1, 1, 1)
    R = torch.sqrt((X - x_g)**2 + (Y - y_g)**2)
    R_max = R.amax(dim=(1, 2), keepdim=True)
    # radii & sigmas
    radii = [f * R_max for f in radii_frac]
    transition_width = transition_frac * R_max

    sigma_base = sigma_base_frac * min(H, W)

    sigmas = [0.0]
    for i in range(len(radii_frac)):
        sigmas.append(sigma_base * (sigma_growth ** i))

    # blurred versions
    blurred_imgs = []
    for sigma in sigmas:
        if sigma == 0:
            blurred_imgs.append(img_tensor)
        else:
            # kernel size automatically derived (maybe change logic?)
            k = int(2 * round(3 * sigma) + 1)
            blurred = TF.gaussian_blur(img_tensor, kernel_size=k, sigma=sigma)
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
        w = torch.exp(-0.5 * ((R - c) / transition_width) ** 2)
        weights.append(w)

    weights = torch.stack(weights, dim=0)
    weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-6)
    
    # weighted blending
    output = torch.zeros_like(img_tensor)

    for w, img_blur in zip(weights, blurred_imgs):
        output += w.unsqueeze(1) * img_blur

    output = output.clamp(0, 255).to(torch.uint8)
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # --- plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # --- Row 1: Inputs & geometry ---
    axes[0, 0].imshow(img)
    axes[0, 0].scatter(x_g.item(), y_g.item(), c="red", s=20)
    axes[0, 0].set_title("Input Frame + Gaze")

    axes[0, 1].imshow(output)
    axes[0, 1].scatter(x_g.item(), y_g.item(), c="red", s=20)
    axes[0, 1].set_title("Blurred Image + Gaze")
    
    axes[0, 2].imshow(np.log(R[0].cpu().numpy() + 1), cmap="inferno")
    axes[0, 2].set_title("Distance R (log)")

    axes[1, 0].imshow(weights[0][0].cpu().numpy(), cmap="viridis")
    axes[1, 0].set_title("Weight: Sharp (σ=0)")

    mid = len(weights) // 2
    axes[1, 1].imshow(weights[mid][0].cpu().numpy(), cmap="viridis")
    axes[1, 1].set_title(f"Weight: Mid (σ={sigmas[mid]})")

    axes[1, 2].imshow(weights[-1][0].cpu().numpy(), cmap="viridis")
    axes[1, 2].set_title(f"Weight: Strong (σ={sigmas[-1]})")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()

    # --- save ---
    file_name = "ego4d_blur_heatmaps_example.png"
    out_path = Path(__file__).resolve().parent / "plots" / "blur_heatmaps" / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {file_name}")   
    
    
def viz_mask_centroids():
    with open(MASK_JSON_PATH, "r") as f:
        data = json.load(f)
    # --- compute all distances ---
    results = []
    for idx, entry in data.items():
        filename = entry["filename"]
        mask = mask_utils.decode(entry["rle"])
        c_raw = compute_centroid(mask)
        c_flood = compute_centroid(fill_mask_holes_floodfill(mask))
        if c_raw is None or c_flood is None:
            continue
        dist = np.linalg.norm(c_raw - c_flood)
        label = FNAME_TO_IDX.get(filename, None)
        class_name = IDX_TO_LABEL.get(label, "unknown") if label is not None else "unknown"
        results.append({
            "mask": mask,
            "mask_filled": fill_mask_holes_floodfill(mask),
            "c_raw": c_raw,
            "c_flood": c_flood,
            "dist": dist,
            "class_name": class_name
        })
    # STATS
    dists = np.array([r["dist"] for r in results])
    mean_dist = dists.mean()
    q95 = np.quantile(dists, 0.95)
    print(f"Mean Δ: {mean_dist:.2f}px")
    print(f"95% quantile: {q95:.2f}px")
    # FILTER TOP OUTLIERS
    filtered = [r for r in results if r["dist"] >= q95]
    # sort descending for nicer plots
    filtered = sorted(filtered, key=lambda x: x["dist"], reverse=True)
    samples = filtered[:4]
    # PLOT
    fig, axes = plt.subplots(2, len(samples), figsize=(4*len(samples), 8))
    for i, r in enumerate(samples):
        # --- raw ---
        ax = axes[0, i]
        ax.imshow(r["mask"], cmap="gray")
        ax.scatter(*r["c_raw"], c="red", s=20)
        ax.scatter(*r["c_flood"], c="blue", s=20)
        ax.set_title(f"{r['class_name']}\nΔ = {r['dist']:.1f}px")
        ax.axis("off")
        if i == 0:
            ax.legend(["raw", "flood"])
        # --- filled ---
        ax = axes[1, i]
        ax.imshow(r["mask_filled"], cmap="gray")
        ax.scatter(*r["c_flood"], c="blue", s=20)
        ax.axis("off")
    plt.tight_layout()
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / "mask_json" / "mask_centroid_outliers.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved → {out_path}")

    
def viz_imagenet_fov_samples():
    crop_fov = setup_exact_foveation("crop")
    blur_fov = setup_exact_foveation("blur-strong")
    cm_fov = setup_exact_foveation("cm-strong")
    val_ds = ImageFolder(IMAGENET_VAL_PATH)
    json_path = "/home/data/elias/imagenet_sam_masks/imagenet_val_gaze_only.json"
    with open(json_path, "r") as f:
        json_data = json.load(f)
    json_by_filename = {
        v["filename"]: v
        for v in json_data.values()
    }
    # --- pick 2 samples ---
    indices = random.sample(range(len(val_ds)), 2)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    col_titles = ["Original", "Crop", "Blur-Strong", "CM-Strong"]
    for row, i in enumerate(indices):
        path, label = val_ds.samples[i]
        filename = Path(path).name
        if filename not in json_by_filename:
            continue
        img = val_ds[i][0]
        img_np = np.array(img)
        dp = json_by_filename[filename]
        img_tensor = pil_to_tensor(img).unsqueeze(0)
        _, _, H, W = img_tensor.shape
        cx = dp["centroid"]["x_rel"] * W
        cy = dp["centroid"]["y_rel"] * H
        gaze = torch.tensor([[cx, cy]], dtype=torch.float32)
        with torch.no_grad():
            crop_img = crop_fov(img_tensor.clone(), gaze, None)
            blur_img = blur_fov(img_tensor.clone(), gaze, None)
            cm_img = cm_fov(img_tensor.clone(), gaze, None)
        def to_np(x):
            return x.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)
        imgs = [
            img_np,
            to_np(crop_img),
            to_np(blur_img),
            to_np(cm_img)
        ]
        for col in range(4):
            axes[row, col].imshow(imgs[col])
            axes[row, col].axis("off")
        # --- gaze only on original ---
        axes[row, 0].scatter(cx, cy, c="red", s=20)
    plt.tight_layout(rect=[0.05, 0, 1, 0.92])

    # column titles
    for col in range(4):
        ax = axes[0, col]
        pos = ax.get_position()
        x = pos.x0 + pos.width / 2
        fig.text(
            x, 0.95,
            col_titles[col],
            ha="center",
            fontsize=12,
            fontweight="bold"
        )
    out_path = Path(__file__).resolve().parent / "plots" / "imagenet_fovs" / "imagenet_examples.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved imagenet_examples.png")
    
    
def viz_imagenet_mask_samples(n):
    
    val_ds = ImageFolder(IMAGENET_VAL_PATH, transform=None)
        
    with open(MASK_JSON_PATH, "r") as f:
        json_data = json.load(f)
        
    json_by_filename = {
        v["filename"]: v
        for v in json_data.values()
    }

    total = len(val_ds)
    #indices = [384, X]
    indices = random.sample(range(total), n)
    
    for i in indices:

        path = val_ds.samples[i][0]
        filename = Path(path).name

        if filename not in json_by_filename:
            print(f"WARNING: {filename} not found in JSON")
            continue

        img = val_ds[i][0]
        W_img, H_img = img.size
        dp = json_by_filename[filename]
        idx = FNAME_TO_IDX.get(filename)
        label = IDX_TO_LABEL.get(idx)
        
        mask = mask_utils.decode(dp["rle"])
        H_mask, W_mask = mask.shape
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (W_img, H_img),
            interpolation=cv2.INTER_NEAREST
        )
        
        # --- Centroids ---
        cx_rel = dp["centroid"]["x_rel"]
        cy_rel = dp["centroid"]["y_rel"]

        cx_abs = cx_rel * W_img
        cy_abs = cy_rel * H_img

        # --- Bounding box ---
        x1, y1, x2, y2 = dp["bbox"]
        
        scale_x = W_img / W_mask
        scale_y = H_img / H_mask
        
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

        # --- Areas ---
        area_mask_rel = dp.get("area_mask_rel", None)
        area_bbox_rel = dp.get("area_bbox_rel", None)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(6,6))

        ax.imshow(img)
        ax.imshow(mask_resized, alpha=0.4)

        # Bounding box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="yellow",
            facecolor="none"
        )
        ax.add_patch(rect)

        # Floodfill centroid (from JSON)
        ax.scatter(cx_abs, cy_abs, c="blue", s=40, label="Centroid")

        title = f"Label: {label}\n"
        if area_mask_rel is not None:
            title += f"Mask area: {area_mask_rel:.3f} | "
        if area_bbox_rel is not None:
            title += f"BBox area: {area_bbox_rel:.3f}"

        ax.set_title(title)
        ax.axis("off")
        ax.legend(loc="lower left")

        plt.tight_layout()
        
        save_name=f"mask_json_debug{i}.png"
        base_dir = Path(__file__).resolve().parent
        out_path = base_dir / "plots" / "mask_json" / save_name

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved {save_name}")
        

def viz_ooc_background_examples():
    root = Path("/home/data/elias/ImageNet-OOC1k_flattened")
    T_pre = v2.Compose([
        v2.Resize(540),
        v2.ToImage(),
        v2.ToDtype(torch.uint8)
    ])
    ds_original = OOCOriginalDataset(root=root, transform=T_pre)
    ds_inpainted = OOCInpaintedDataset(root=root, transform=T_pre)
    # --- pick 3 samples ---
    indices = random.sample(range(len(ds_original)), 3)
    # --- layout: 1 row, 6 columns ---
    fig, axes = plt.subplots(1, 6, figsize=(12, 3))
    def to_np(img):
        if torch.is_tensor(img):
            return img.permute(1, 2, 0).cpu().numpy()
        return np.array(img)
    for i, idx in enumerate(indices):
        col = i * 2
        img_orig, label, gaze = ds_original[idx]
        img_bg, _, _ = ds_inpainted[idx]
        img_orig = to_np(img_orig)
        img_bg = to_np(img_bg)
        # --- original ---
        axes[col].imshow(img_orig)
        axes[col].axis("off")
        # --- background only ---
        axes[col + 1].imshow(img_bg)
        axes[col + 1].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    fig.text(
        0.5, 0.92,
        "Original vs Background-Only",
        ha="center",
        fontsize=13,
        fontweight="bold"
    )
    # --- save ---
    out_path = Path(__file__).resolve().parent / "plots" / "ooc" / "ooc_background_examples.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved ooc_background_examples.png")
    
    
def viz_full_imagenet_ooc():
    modes = {
        "original": "Original",
        "object": "Object-Only",
        "ooc": "Out-of-Context"
    }
    mode_keys = list(modes.keys())
    datasets = {
        m: ImageNetValOOC(mode=m, seed=42, transform=None)
        for m in mode_keys
    }
    # --- random indices ---
    total = len(next(iter(datasets.values())))
    indices = random.sample(range(total), 2)
    # --- figure ---
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for row, idx in enumerate(indices):
        for col, mode in enumerate(mode_keys):
            ds = datasets[mode]
            img, label, gaze = ds[idx]
            img = np.array(img)
            H, W = img.shape[:2]
            gx = gaze[0].item() * W
            gy = gaze[1].item() * H
            ax = axes[row, col]
            ax.imshow(img)
            ax.scatter(gx, gy, c="red", s=20)
            ax.axis("off")
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    fig.text(
        0.5, 0.97,
        "Out-of-Context Evaluation Examples",
        ha="center",
        fontsize=13,
        fontweight="bold"
    )
    # --- column titles ---
    for col, mode in enumerate(mode_keys):
        pos = axes[0, col].get_position()
        x = pos.x0 + pos.width / 2
        y = pos.y1 + 0.015 
        fig.text(
            x, y,
            modes[mode],
            ha="center",
            fontsize=11,
            fontweight="bold"
        )
    save_name = "full_ooc_examples.png"
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / "ooc" / save_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {save_name}")
    

def viz_crowding_overview():
    conditions = ["a", "ax", "xax"]
    fov_models = ["crop", "blur-nosal", "cm-strong"]
    fov_names = {
        "crop": "Crop",
        "blur-nosal": "Blur-Medium",
        "cm-strong": "CM-Strong"
    }
    # --- SAME index for all ---
    base_ds = CrowdingDataset(transform=PILToTensor(), condition="a")
    #idx = random.randint(0, len(base_ds) - 1)
    idx = 737
    # --- datasets ---
    datasets = {
        c: CrowdingDataset(transform=PILToTensor(), condition=c)
        for c in conditions
    }
    # --- foveations ---
    fovs = {
        m: setup_exact_foveation(m)
        for m in fov_models
    }
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    # ROW 1: CONDITIONS (RAW)
    for col, cond in enumerate(conditions):
        img, label, gaze = datasets[cond][idx]
        img_np = img.permute(1, 2, 0).numpy()
        H, W = img_np.shape[:2]
        gx = gaze[0].item() * W
        gy = gaze[1].item() * H
        ax = axes[0, col]
        ax.imshow(img_np)
        ax.axis("off")
    # ROW 2: FOVEATION (XAX)
    img_np = img.permute(1, 2, 0).numpy()
    H, W = img_np.shape[:2]
    gx = gaze[0].item() * W
    gy = gaze[1].item() * H
    for col, m in enumerate(fov_models):
        fov = fovs[m]
        img_tensor = img.unsqueeze(0).float()
        gaze_abs = torch.tensor([[gx, gy]])
        with torch.no_grad():
            img_fov = fov(img_tensor, gaze_abs, None)
        out = (
            img_fov.squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        ax = axes[1, col]
        ax.imshow(out)
        ax.axis("off")
    # COLUMN TITLES
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.9])
    col_titles = ["a", "ax", "xax"]
    for col, title in enumerate(col_titles):
        pos = axes[0, col].get_position()
        x = pos.x0 + pos.width / 2
        y = pos.y1 + 0.015
        fig.text(x, y, title, ha="center", fontsize=11, fontweight="bold")
    # =========================
    # FOVEATION LABELS (ROW 2)
    # =========================
    for col, m in enumerate(fov_models):
        pos = axes[1, col].get_position()
        x = pos.x0 + pos.width / 2
        y = pos.y0 - 0.03   # ← Abstand unter Bild (tweakbar!)
        fig.text(x, y, fov_names[m], ha="center", fontsize=10, fontweight="bold")
    # ROW LABELS
    for row, name in enumerate(["Conditions", "Foveation (xax)"]):
        pos = axes[row, 0].get_position()
        y = pos.y0 + pos.height / 2
        fig.text(0.02, y, name, va="center", rotation=90, fontsize=11)
    # GLOBAL TITLE
    fig.text(
        0.5, 0.95,
        "Crowding Setup and Foveation Effect",
        ha="center",
        fontsize=13,
        fontweight="bold"
    )
    save_name = "crowding_overview.png"
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / "crowding" / save_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {save_name}")
    

def load_sample(i):
    ANNOT_PATH = "/home/data/elias/Ego4dDivSubset/annot.parquet"
    H5_PATH = "/home/data/elias/Ego4dDivSubset/ego4d_diverse_subset.h5"

    df = pd.read_parquet(ANNOT_PATH)

    with h5py.File(H5_PATH, "r") as hf:
        
        # print({k: hf[k].shape for k in hf.keys()})
        frame = hf.get("frames")[i]
        saliency = hf.get("saliency")[i]
        frame = Image.open(io.BytesIO(frame)).convert("RGB")
        # convert from BGR to RGB
        frame = np.array(frame)[:, :, ::-1]
        frame = Image.fromarray(frame)
    
    annot = df.iloc[i]
    saliency = saliency.astype(np.float32)
    
    return frame, annot, saliency


def prepare_tensors(frame, annot, saliency):

    img_np = np.array(frame)
    H, W, _ = img_np.shape

    img_tensor = (
        torch.from_numpy(img_np)
        .permute(2,0,1)
        .unsqueeze(0)
        .to(device)
        .to(torch.float32)
    )

    gaze_tensor = torch.tensor(
        [[annot.gaze_loc_x, annot.gaze_loc_y]],
        device=device
    ).float()
    
    S = cv2.resize(saliency, (W,H))
    S = (S - S.min()) / (S.max() - S.min() + 1e-6)

    sal_tensor = (
        torch.from_numpy(S)
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    return img_tensor, gaze_tensor, sal_tensor


if __name__ == "__main__":
    main()