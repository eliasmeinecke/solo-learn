
import io
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from foveation.gaze_crop import GazeCenteredCrop


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


def viz_fov(df, index, frame, method):
    
    row = df.iloc[index]

    if method == "crop":
        crop = GazeCenteredCrop(crop_size=240)
        out = crop(frame, row)
    elif method == "blur":  # to be implemented
        pass  
    elif method == "log_polar":  # to be implemented
        pass
    else:
        out = frame

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(frame)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Foveation")
    plt.imshow(out)
    plt.axis("off")

    plt.tight_layout()

    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / method / "ego4d_crop_example.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)

    print("Saved crop_example.png")    


if __name__ == "__main__":
    ANNOT_PATH = "/home/data/elias/Ego4dDivSubset/annot.parquet"
    H5_PATH = "/home/data/elias/Ego4dDivSubset/ego4d_diverse_subset.h5"

    i = 100_001

    df = pd.read_parquet(ANNOT_PATH)

    with h5py.File(H5_PATH, "r") as hf:
        frame = hf.get("frames")[i]
        saliency = hf.get("saliency")[i]

        frame = Image.open(io.BytesIO(frame))
    
    # methods: crop, blur, log_polar, ...
    method = "crop"

    # viz_saliency(df, i, frame, saliency)
    viz_fov(df, i, frame, method)