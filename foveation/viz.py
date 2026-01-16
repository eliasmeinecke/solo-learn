
import io
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from foveation.gaze_crop import GazeCenteredCrop
from foveation.radial_blur import RadialBlurFoveation
from foveation.fcg import FovealCartesianGeometry


def main():
    ANNOT_PATH = "/home/data/elias/Ego4dDivSubset/annot.parquet"
    H5_PATH = "/home/data/elias/Ego4dDivSubset/ego4d_diverse_subset.h5"

    i = 100_003

    df = pd.read_parquet(ANNOT_PATH)

    with h5py.File(H5_PATH, "r") as hf:
        frame = hf.get("frames")[i]
        saliency = hf.get("saliency")[i]
        frame = Image.open(io.BytesIO(frame)).convert("RGB")
        # convert from BGR to RGB
        frame = np.array(frame)[:, :, ::-1]
        frame = Image.fromarray(frame)
    
    # methods: crop, blur, fcg
    viz_fov(df, i, frame, "crop")
    viz_fov(df, i, frame, "blur")
    viz_fov(df, i, frame, "fcg")
    
    # viz_fcg_rings()
    viz_saliency(df, i, frame, saliency)


def viz_fov(df, index, frame, method):
    
    row = df.iloc[index]
    x_g, y_g = row.gaze_loc_x, row.gaze_loc_y

    if method == "crop":
        out = GazeCenteredCrop()(frame, row)
    elif method == "blur":  # wip
        out = RadialBlurFoveation()(frame, row)
    elif method == "fcg":  # wip
        out = FovealCartesianGeometry()(frame, row)
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

    print(f"Saved {file_name}")  
    

def viz_fcg_rings():
    # visualizes "mapping onto rings of original image"
    fcg = FovealCartesianGeometry(p0=15, pmax=100, nR=30)
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

    print(f"Saved {file_name}")   


def viz_saliency(df, index, frame, saliency):
    row = df.iloc[index]
    flat_max = saliency.argmax()
    max_x, max_y = flat_max % 64, flat_max // 64
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.array(frame))
    ax[1].imshow(saliency)
    ax[0].axis("off")
    ax[1].axis("off")
    ax[0].scatter(row.gaze_loc_x, row.gaze_loc_y, c="red", s=50)
    ax[1].scatter(max_x, max_y, c="red", s=50)
    plt.tight_layout()
    
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / "ego4d" / "ego4d_example.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print("Saved ego4d_example.png")  


if __name__ == "__main__":
    main()