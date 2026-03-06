
import io
import time
import numpy as np
import pandas as pd
import h5py
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from types import SimpleNamespace
import torch
import torchvision.transforms.functional as TF

from foveation.factory import GazePredictor
from foveation.methods.radial_blur import RadialBlurFoveation
from foveation.methods.cm import CortalMagnification, radial_quadratic_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    indices = [99_002, 100_003]

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
    
    viz_fov(samples, method="blur")
    
    # needs changing after gpu-switch:
    # viz_cm_saliency_effect(samples)
    
    # clean up/repair later:
    # viz_blur_heatmaps(frame, annot, saliency)
    # viz_saliency(frame, annot, saliency)
    # viz_eval_saliency(frame)
    
    # update/create & test:
    # viz_relative_sigmas(frame, annot, saliency)
    # viz_relative_cm(frame, annot, saliency)
    


def viz_fov(samples, method="cm"):

    if method == "blur":
        fov = RadialBlurFoveation().to(device)
    elif method == "cm":
        fov = CortalMagnification().to(device)

    n = len(samples)

    fig, axes = plt.subplots(n, 2, figsize=(8,4*n))

    for i, sample in enumerate(samples):

        frame = sample["frame"]
        img_tensor = sample["img_tensor"]
        sal_tensor = sample["sal_tensor"]
        gaze_tensor = sample["gaze_tensor"]
        annot = sample["annot"]

        with torch.no_grad():
            out_tensor = fov(img_tensor, gaze_tensor, sal_tensor)

        out = (
            out_tensor.squeeze(0)
            .permute(1,2,0)
            .cpu()
            .numpy()
        )

        axes[i,0].imshow(frame)
        axes[i,0].scatter(
            annot.gaze_loc_x,
            annot.gaze_loc_y,
            c="red",
            s=20
        )
        axes[i,0].set_title("Original")
        axes[i,0].axis("off")

        axes[i,1].imshow(out)
        axes[i,1].scatter(
            annot.gaze_loc_x,
            annot.gaze_loc_y,
            c="red",
            s=20
        )
        axes[i,1].set_title(method)
        axes[i,1].axis("off")

    plt.tight_layout()

    file_name = f"ego4d_{method}_example.png"
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / method / file_name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {file_name}")  
    
    
def viz_relative_sigmas(frame, annot, saliency):
    
    method = "blur"
    
    img_np = np.array(frame)
    H, W, _ = img_np.shape
    x_g, y_g = annot.gaze_loc_x, annot.gaze_loc_y

    s_big = cv2.resize(saliency, (W, H), interpolation=cv2.INTER_LINEAR)
    s_big = (s_big - s_big.min()) / (s_big.max() - s_big.min() + 1e-6)
    big_out = RadialBlurFoveation()(frame, annot, s_big)
    
    scale_x = 224 / W
    scale_y = 224 / H

    small_annot = SimpleNamespace(
        gaze_loc_x = x_g * scale_x,
        gaze_loc_y = y_g * scale_y
    )
    small_img_np = np.array(frame.resize((224, 224), resample=Image.BILINEAR))
    s_small = cv2.resize(saliency, (224, 224), interpolation=cv2.INTER_LINEAR)
    s_small = (s_small - s_small.min()) / (s_small.max() - s_small.min() + 1e-6)
    small_out = RadialBlurFoveation()(small_img_np, small_annot, s_small)
    # small_out = small_out.resize((H, W), resample=Image.BILINEAR)
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("540x540")
    plt.imshow(big_out)
    plt.scatter(x_g, y_g, c="red", s=20)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("224x224")
    plt.imshow(small_out)
    # plt.scatter(x_g, y_g, c="red", s=20)
    plt.scatter(small_annot.gaze_loc_x, small_annot.gaze_loc_y, c="red", s=20)
    plt.axis("off")

    plt.tight_layout()

    file_name = f"ego4d_{method}_example.png"
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / method / file_name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {file_name}")  


def viz_blur_heatmaps(frame, annot, saliency):
    radii_frac = [0.3, 0.7]
    sigma_base_frac = 0.006
    sigma_growth = 2
    saliency_alpha = 5 
    transition_frac = 0.1

    img_np = np.array(frame)
    H, W, _ = img_np.shape
        
    x_g, y_g = annot.gaze_loc_x, annot.gaze_loc_y

    # --- saliency ---
    S = cv2.resize(saliency, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    # --- coordinate grid ---
    ys = np.arange(H)
    xs = np.arange(W)
    X, Y = np.meshgrid(xs, ys)

    # --- distances ---
    R = np.sqrt((X - x_g)**2 + (Y - y_g)**2)
    R_max = np.max(R)
    R_eff = R / (1.0 + saliency_alpha * S)
    
    radii = [f * R_max for f in radii_frac]
    transition_width = transition_frac * R_max
    sigma_base = sigma_base_frac * 540
    sigmas = [0]
    for i in range(len(radii_frac)):
        sigmas.append(sigma_base * (sigma_growth ** i)) 

    # --- ring centers ---
    ring_centers = []
    prev = 0.0
    for r in radii:
        ring_centers.append(0.5 * (prev + r))
        prev = r
    ring_centers.append(prev + transition_width)

    # --- gaussian weights ---
    weights = []
    for c in ring_centers:
        w = np.exp(-0.5 * ((R_eff - c) / transition_width) ** 2)
        weights.append(w)

    weights = np.stack(weights, axis=0)
    weight_sum = np.sum(weights, axis=0)
    weights /= weight_sum + 1e-6

    # --- plotting ---
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # --- Row 1: Inputs & geometry ---
    axes[0, 0].imshow(img_np)
    axes[0, 0].scatter(x_g, y_g, c="red", s=30)
    axes[0, 0].set_title("Input Frame + Gaze")

    axes[0, 1].imshow(np.log(R + 1), cmap="inferno")
    axes[0, 1].set_title("Distance R (log)")

    axes[0, 2].imshow(weights[0], cmap="viridis")
    axes[0, 2].set_title("Weight: Sharp (σ=0)")

    axes[0, 3].imshow(weights[-1], cmap="viridis")
    axes[0, 3].set_title(f"Weight: Strong (σ={sigmas[-1]})")

    # --- Row 2: Attention & effects ---
    axes[1, 0].imshow(S, cmap="magma")
    axes[1, 0].set_title("Saliency Map")

    axes[1, 1].imshow(np.log(R_eff + 1), cmap="inferno")
    axes[1, 1].set_title("Effective Distance R_eff (log)")

    mid = len(weights) // 2
    axes[1, 2].imshow(weights[mid], cmap="viridis")
    axes[1, 2].set_title(f"Weight: Mid (σ={sigmas[mid]})")

    diff = R - R_eff
    axes[1, 3].imshow(diff, cmap="coolwarm")
    axes[1, 3].set_title("Distance Reduction (R − R_eff)")

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


def viz_cm_saliency_effect(samples, betas=[0.0, 1.0, 3.0]):
    
    sample = samples[0]
    frame = sample["frame"]
    img_tensor = sample["img_tensor"]
    sal_tensor = sample["sal_tensor"]
    gaze_tensor = sample["gaze_tensor"]
        
    device = img_tensor.device
    B, C, H, W = img_tensor.shape

    x_g = gaze_tensor[0,0].item()
    y_g = gaze_tensor[0,1].item()

    with torch.no_grad():

        ys = torch.arange(H, device=device).view(1, H, 1).expand(1, H, W)
        xs = torch.arange(W, device=device).view(1, 1, W).expand(1, H, W)

        dx = xs - x_g
        dy = ys - y_g
        r = torch.sqrt(dx**2 + dy**2 + 1e-6)

        sal_norm = sal_tensor / (
            sal_tensor.sum(dim=(2,3), keepdim=True) + 1e-6
        )
        sal_norm = sal_norm.squeeze(1)

        mean_r = torch.sum(sal_norm * r, dim=(1,2), keepdim=True)
        mean_r2 = torch.sum(sal_norm * r**2, dim=(1,2), keepdim=True)
        std_r = torch.sqrt(torch.clamp(mean_r2 - mean_r**2, min=0.0))
        spread_norm = std_r / r.amax(dim=(1,2), keepdim=True)
        spread_value = spread_norm.item()

    base_size = min(H, W)
    fov_base = 0.3125 * base_size
    K = 0.208 * base_size

    # -------------------------------------------------
    # Plot Layout: 2 rows × 4 columns
    # -------------------------------------------------

    fig, axes = plt.subplots(2, 4, figsize=(18,9))

    # -----------------------
    # Row 1: Original + CM
    # -----------------------

    axes[0,0].imshow(frame)
    axes[0,0].scatter(x_g, y_g, c="red", s=20)
    axes[0,0].set_title("Original")
    axes[0,0].axis("off")

    # -----------------------
    # Row 2: Saliency
    # -----------------------

    sal_np = sal_tensor.squeeze().cpu().numpy()
    axes[1,0].imshow(sal_np, cmap="viridis")
    axes[1,0].scatter(x_g, y_g, c="red", s=20)
    axes[1,0].set_title(f"Saliency\nspread_norm={spread_value:.3f}")
    axes[1,0].axis("off")

    # -----------------------
    # CM variants
    # -----------------------

    for col, beta in enumerate(betas):

        cm = CortalMagnification(saliency_beta=beta).to(device)

        with torch.no_grad():
            out_tensor = cm(img_tensor, gaze_tensor, sal_tensor)

        out_np = (
            out_tensor.squeeze(0)
            .permute(1,2,0)
            .cpu()
            .numpy()
        )

        fov_eff = fov_base * (1 + beta * spread_value)

        # ---- CM Image ----
        axes[0, col+1].imshow(out_np.astype(np.uint8))
        axes[0, col+1].scatter(x_g, y_g, c="red", s=20)
        axes[0, col+1].set_title(
            f"CM β={beta}\nfov_eff={fov_eff:.1f}"
        )
        axes[0, col+1].axis("off")

        # ---- Distortion Map ----
        fov_eff_tensor = torch.tensor(
            fov_base * (1 + beta * spread_value),
            device=device
        )

        r_new = radial_quadratic_batch(
            r,
            fov_eff_tensor.view(1,1,1),
            torch.tensor(K, device=device).view(1,1,1)
        )
        
    
        eps = 1e-6
        magnification = (r_new / (r + eps)).squeeze().cpu().numpy()

        # robuste Kontrast-Skalierung
        vmin = np.percentile(magnification, 2)
        vmax = np.percentile(magnification, 98)

        im = axes[1, col+1].imshow(
            magnification,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax
        )

        axes[1, col+1].scatter(x_g, y_g, c="white", s=20)
        axes[1, col+1].set_title("Magnification (r_new / r)")
        axes[1, col+1].axis("off")


    plt.tight_layout()

    # --- save ---
    file_name = "cm_saliency_effect.png"
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / "cm_saliency" / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {file_name}")


def viz_saliency(frame, annot, saliency):
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
    
    
def viz_eval_saliency(frame):
    """
    Visualize GazePredictor saliency output for a single frame.

    Args:
        frame (PIL.Image or np.ndarray): RGB image
        save_name (str): output filename
    """

    # Prepare image
    if isinstance(frame, Image.Image):
        img_np = np.array(frame)
    else:
        img_np = frame

    H, W = img_np.shape[:2]

    # Convert to tensor (B,C,H,W)
    img_tensor = (
        torch.from_numpy(img_np)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(torch.uint8)
    )

    # Run GazePredictor
    gaze_predictor = GazePredictor()
    gaze, saliency = gaze_predictor(img_tensor)

    # Remove batch dimension
    gaze = gaze[0]
    saliency = saliency[0, 0].cpu().numpy()

    gaze_x, gaze_y = gaze.tolist()

    # Find saliency maximum
    flat_max = saliency.argmax()
    max_y, max_x = np.unravel_index(flat_max, saliency.shape)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Image + gaze
    ax[0].imshow(img_np)
    ax[0].scatter(gaze_x, gaze_y, c="red", s=60)
    ax[0].set_title("Image + Predicted Gaze")
    ax[0].axis("off")

    # Saliency
    ax[1].imshow(saliency, cmap="viridis")
    ax[1].scatter(max_x, max_y, c="red", s=60)
    ax[1].set_title("Saliency Map + Maximum")
    ax[1].axis("off")

    plt.tight_layout()

    save_name="saliency_eval_debug.png"
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / "eval_saliency" / save_name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {save_name}")
    

def load_sample(i):
    ANNOT_PATH = "/home/data/elias/Ego4dDivSubset/annot.parquet"
    H5_PATH = "/home/data/elias/Ego4dDivSubset/ego4d_diverse_subset.h5"

    df = pd.read_parquet(ANNOT_PATH)

    with h5py.File(H5_PATH, "r") as hf:
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
        .to(torch.uint8)
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