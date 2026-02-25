
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

from foveation.factory import FoveationTransform
from foveation.methods.gaze_crop import GazeCenteredCrop
from foveation.methods.radial_blur import RadialBlurFoveation
from foveation.methods.cm import CortalMagnification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    ANNOT_PATH = "/home/data/elias/Ego4dDivSubset/annot.parquet"
    H5_PATH = "/home/data/elias/Ego4dDivSubset/ego4d_diverse_subset.h5"

    # i = 99_002 # (for 2 saliency "blobs")
    i = 100_003

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
     
    # updated after gpu-switch:
    viz_fov(frame, annot, saliency, "blur")
    viz_fov(frame, annot, saliency, "cm") # methods: crop, blur, cm
    # viz_eval_saliency(frame)
    
    # needs changing after gpu-switch: (maybe pull gaze & saliency tensors to main?)
    # benchmark_fov(frame, annot, saliency, "cm")
    # viz_relative_sigmas(frame, annot, saliency)
    # viz_blur_heatmaps(frame, annot, saliency)
    # viz_cm_overview(frame, annot, saliency)
    # viz_saliency(frame, annot, saliency)


def viz_fov(frame, annot, saliency, method):

    img_np = np.array(frame)
    H, W, _ = img_np.shape

    x_g, y_g = annot.gaze_loc_x, annot.gaze_loc_y

    if method == "crop":
        out = GazeCenteredCrop()(frame, annot)
    elif method in ["blur", "cm"]: # GPU Foveations

        # ---- Image → Tensor ----
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # C,H,W
        img_tensor = img_tensor.unsqueeze(0).to(device)  # B,C,H,W
        img_tensor = img_tensor.to(torch.uint8)

        # ---- Saliency ----
        S = cv2.resize(saliency, (W, H), interpolation=cv2.INTER_LINEAR)
        S = (S - S.min()) / (S.max() - S.min() + 1e-6)

        sal_tensor = torch.from_numpy(S).unsqueeze(0).unsqueeze(0)
        sal_tensor = sal_tensor.to(device).float()

        gaze_tensor = torch.tensor([[x_g, y_g]], device=device).float()

        if method == "blur":
            fov = RadialBlurFoveation().to(device)
        else:
            fov = CortalMagnification().to(device)
        with torch.no_grad():
            out_tensor = fov(img_tensor, gaze_tensor, sal_tensor)

        out_tensor = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = Image.fromarray(out_tensor.astype(np.uint8))
    else:
        out = frame

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(frame)
    plt.scatter(x_g, y_g, c="red", s=20)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Foveation")
    plt.imshow(out)
    plt.scatter(x_g, y_g, c="red", s=20)
    plt.axis("off")

    plt.tight_layout()

    file_name = f"ego4d_{method}_example.png"
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / method / file_name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {file_name}")  
    
    
def benchmark_fov(frame, annot, saliency, method, runs=100):
    
    if method == "crop":
        foveation = GazeCenteredCrop()
    elif method == "blur":
        foveation = RadialBlurFoveation()
    elif method == "cm":  # wip
        foveation = CortalMagnification()
    else:
        print("No benchmarkable foveation given.")
        return None

    for _ in range(10):
        _ = foveation(frame, annot, saliency)
    
    start = time.perf_counter()
    
    for _ in range(runs):
        _ = foveation(frame, annot, saliency)
    
    end = time.perf_counter()
    
    avg_time = (end - start) / runs
    
    print(f"Using fov method: {method}")
    print(f"Average fov time per image: {avg_time:.4f} seconds")
    print(f"Images per second: {1/avg_time:.2f}")
    
    
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


def viz_cm_overview(frame, annot, saliency):

    img_np = np.array(frame)
    H, W, _ = img_np.shape

    x_g, y_g = annot.gaze_loc_x, annot.gaze_loc_y

    # --- normalize saliency ---
    S = cv2.resize(saliency, (W, H), interpolation=cv2.INTER_LINEAR)
    S = S.astype(np.float32)
    S = (S - S.min()) / (S.max() - S.min() + 1e-6)

    # --- CM instances ---
    cm_no_sal = CortalMagnification(saliency_beta=0.0)
    cm_sal = CortalMagnification(saliency_beta=1)

    out_no_sal = cm_no_sal(frame, annot, S)
    out_sal = cm_sal(frame, annot, S)

    # --- plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original
    axes[0, 0].imshow(frame)
    axes[0, 0].scatter(x_g, y_g, c="red", s=30)
    axes[0, 0].set_title("Original Image + Gaze")

    # CM without saliency
    axes[0, 1].imshow(out_no_sal)
    axes[0, 1].scatter(x_g, y_g, c="red", s=30)
    axes[0, 1].set_title("Cortical Magnification")

    # Saliency
    axes[1, 0].imshow(S, cmap="magma")
    axes[1, 0].set_title("Saliency Map")

    # CM with saliency
    axes[1, 1].imshow(out_sal)
    axes[1, 1].scatter(x_g, y_g, c="red", s=30)
    axes[1, 1].set_title("CM + Saliency")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()

    # --- save ---
    file_name = "ego4d_cm_overview.png"
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / "cm_overview" / file_name
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

    foveation_transform = FoveationTransform(foveation=None, base_transform=lambda x: x)
    
    annot = foveation_transform._build_center_annotation(frame)
    saliency = foveation_transform._build_center_saliency()
    
    flat_max = saliency.argmax()
    max_x, max_y = flat_max % 64, flat_max // 64

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    # original
    ax[0].imshow(np.array(frame))
    ax[0].scatter(annot.gaze_loc_x, annot.gaze_loc_y, c="red", s=50)
    ax[0].set_title("Image + Gaze")
    ax[0].axis("off")

    # saliency
    ax[1].imshow(saliency, cmap="viridis")
    ax[1].scatter(max_x, max_y, c="red", s=50)
    ax[1].set_title("Saliency + Max")
    ax[1].axis("off")

    plt.tight_layout()

    save_name="saliency_eval_debug.png"
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / "eval_saliency" / save_name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {save_name}")
    

if __name__ == "__main__":
    main()