
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

from foveation.factory import FoveationTransform
from foveation.methods.gaze_crop import GazeCenteredCrop
from foveation.methods.radial_blur import RadialBlurFoveation
from foveation.methods.fcg import FovealCartesianGeometry
from foveation.methods.fcg_saliency import FovealCartesianGeometryWithSaliency
from foveation.methods.fcg_paper import FovealCartesianGeometryPaper



def main():
    ANNOT_PATH = "/home/data/elias/Ego4dDivSubset/annot.parquet"
    H5_PATH = "/home/data/elias/Ego4dDivSubset/ego4d_diverse_subset.h5"

    #i = 99_002 (for 2 saliency "blobs")
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
    
    # also test?: fcg_paper, fcg_saliency, cortal_magnification
     
    # methods: crop, blur, fcg
    # viz_fov(frame, annot, saliency, "blur")
    # performance test:
    # benchmark_blur(frame, annot, saliency, "blur")
    
    # viz_blur_heatmaps(frame, annot, saliency)
    # viz_relative_sigmas(frame, annot, saliency)
    
    # viz_fcg_grids(frame, annot, saliency)
    # viz_fcg_rings_paper()
    
    # viz_saliency(frame, annot, saliency)
    # viz_eval_saliency(frame, FoveationTransform(foveation=None, base_transform=lambda x: x))


def benchmark_blur(frame, annot, saliency, method, runs=100):
    
    if method == "crop":
        foveation = GazeCenteredCrop()
    elif method == "blur":
        foveation = RadialBlurFoveation()(frame, annot, saliency)
    elif method == "fcg":  # wip
        foveation = FovealCartesianGeometry()
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
    
    print(f"Average blur time per image: {avg_time:.4f} seconds")
    print(f"Images per second: {1/avg_time:.2f}")
    

def viz_fov(frame, annot, saliency, method):
    
    img_np = np.array(frame)
    H, W, _ = img_np.shape

    x_g, y_g = annot.gaze_loc_x, annot.gaze_loc_y
    S = cv2.resize(saliency, (W, H), interpolation=cv2.INTER_LINEAR)
    S = (S - S.min()) / (S.max() - S.min() + 1e-6)

    print(f"{method} input size: {frame.size}")
    
    if method == "crop":
        out = GazeCenteredCrop()(frame, annot, S)
        out.resize((540, 540), resample=Image.BILINEAR)
    elif method == "blur":  # wip
        out = RadialBlurFoveation()(frame, annot, S)
    elif method == "fcg":  # wip
        out = FovealCartesianGeometry()(frame, annot, S)
        out.resize((540, 540), resample=Image.BILINEAR)
    else:
        out = frame
    
    print(f"{method} output size: {out.size}")

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
    radii_frac = [0.2, 0.4, 0.8]
    sigma_base = 0.5 
    sigma_growth = 2
    saliency_alpha = 1.0 
    transition_frac = 0.1

    img_np = np.array(frame)
    H, W, _ = img_np.shape
        
    x_g, y_g = annot.gaze_loc_x, annot.gaze_loc_y

    # --- saliency ---
    S = cv2.resize(saliency, (W, H), interpolation=cv2.INTER_LINEAR)
    S = S.astype(np.float32)
    S = (S - S.min()) / (S.max() - S.min() + 1e-6)

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
    sigmas = [sigma_base * (sigma_growth ** i) for i in range(len(radii_frac)+1)]

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
    
    
def viz_fcg_grids(frame, annot, saliency):

    # --- prepare fcg with and without saliency ---
    fcg_no_sal = FovealCartesianGeometry(p0=32, alpha=0.5, beta=0.0)
    fcg_sal    = FovealCartesianGeometry(p0=32, alpha=0.5, beta=1.0)
    
    # --- prepare saliency ---
    img_np = np.array(frame)
    H, W, _ = img_np.shape

    S = cv2.resize(saliency, (W, H), interpolation=cv2.INTER_LINEAR)
    S = (S - S.min()) / (S.max() - S.min() + 1e-6)

    
    # --- apply FCG to real image ---
    img_no_sal = fcg_no_sal(frame, annot, S)
    img_sal    = fcg_sal(frame, annot, S)

    # --- grid visualization ---
    grid = make_grid_image(min(H, W), step=24)
    grid_no_sal = fcg_no_sal(grid, annot, S)
    grid_sal    = fcg_sal(grid, annot, S)

    # --- plotting ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # row 1: real image
    axes[0, 0].imshow(frame)
    axes[0, 0].scatter(annot.gaze_loc_x, annot.gaze_loc_y, c="red", s=30)
    axes[0, 0].set_title("Input + Gaze")

    axes[0, 1].imshow(img_no_sal)
    axes[0, 1].set_title("FCG (no saliency)")

    axes[0, 2].imshow(img_sal)
    axes[0, 2].set_title("FCG (with saliency)")

    # row 2: geometry
    axes[1, 0].imshow(S, cmap="magma")
    axes[1, 0].set_title("Saliency Map")
    
    axes[1, 1].imshow(grid_no_sal)
    axes[1, 1].set_title("Geometry (no saliency)")

    axes[1, 2].imshow(grid_sal)
    axes[1, 2].set_title("Geometry (with saliency)")

    for ax in axes.flat:
        ax.axis("off")

    # --- save ---
    file_name = "ego4d_fcg_grids_example.png"
    out_path = Path(__file__).resolve().parent / "plots" / "fcg_grids" / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {file_name}") 
    
    
def make_grid_image(size, step=16):
    img = np.ones((size, size, 3), dtype=np.uint8) * 255

    for y in range(0, size, step):
        img[y:y+1, :, :] = 0
    for x in range(0, size, step):
        img[:, x:x+1, :] = 0
    
    return Image.fromarray(img)
    

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
    out_path = base_dir / "plots" / "ego4d" / "ego4d_example.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved ego4d_example.png")  
    
    
def viz_eval_saliency(frame, foveation_transform):

    annot = foveation_transform._build_center_annotation(frame)
    saliency = foveation_transform._build_center_saliency(frame)

    H, W = saliency.shape
    
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
    out_path = base_dir / "plots" / "debug" / save_name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {save_name}")
    
    
def viz_fcg_rings_paper():
    # visualizes "mapping onto rings of original image"
    fcg = FovealCartesianGeometryPaper(p0=15, pmax=100, nR=30)
    canvas = np.zeros((fcg.fovea_size, fcg.fovea_size), dtype=np.int32)

    R = 200  # toy image radius

    for y in range(-R, R):
        for x in range(-R, R):
            xp, yp = fcg.mapping(x, y)
            if 0 <= xp < fcg.fovea_size and 0 <= yp < fcg.fovea_size:
                canvas[yp, xp] += 1

    plt.figure(figsize=(6, 6))
    plt.imshow(canvas, cmap="gray")
    plt.title("FCG mapping → foveal rings")
    plt.axis("off")
    
    file_name = f"ego4d_fcg_map_example.png"
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / "fcg_map" / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {file_name}")   


if __name__ == "__main__":
    main()