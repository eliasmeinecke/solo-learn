import io
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import PILToTensor

from foveation.methods.gaze_crop import GazeCenteredCropGPU
from foveation.methods.radial_blur import RadialBlurFoveation
from foveation.methods.cm import CorticalMagnification

DATASET_PATH = "/home/data/ILSVRC_real/val"
GAZE_JSON = "/home/data/elias/imagenet_sam_masks/imagenet_val_masks_with_center.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def benchmark_foveations(n):

    crop_fov = GazeCenteredCropGPU().to(DEVICE)
    blur_fov = RadialBlurFoveation().to(DEVICE)
    cm_fov = CorticalMagnification().to(DEVICE)

    val_ds = ImageFolder(DATASET_PATH, transform=None)

    with open(GAZE_JSON) as f:
        json_data = json.load(f)

    json_by_filename = {v["filename"]: v for v in json_data.values()}

    indices = random.sample(range(len(val_ds)), n)

    pil_to_tensor = PILToTensor()

    timings = {
        "pil_to_tensor": [],
        "crop_torch": [],
        "blur_torch": [],
        "cm_torch": [],
        "tensor_to_pil": [],
    }

    for i in indices:

        path = val_ds.samples[i][0]
        filename = Path(path).name

        if filename not in json_by_filename:
            continue

        img = val_ds[i][0]
        
        dp = json_by_filename[filename]
        W, H = img.size
        
        cx = dp["centroid"]["x_rel"] * W
        cy = dp["centroid"]["y_rel"] * H

        gaze = torch.tensor([[cx, cy]], dtype=torch.float32, device=DEVICE)
        
        # -----------------------------
        # PIL → Tensor
        # -----------------------------
        t0 = time.perf_counter()
        img_tensor = pil_to_tensor(img).unsqueeze(0).to(DEVICE)
        timings["pil_to_tensor"].append(time.perf_counter() - t0)

        # -----------------------------
        # Tensor foveations
        # -----------------------------
        with torch.no_grad():

            torch.cuda.synchronize() if DEVICE == "cuda" else None
            t0 = time.perf_counter()
            crop_fov(img_tensor.clone(), gaze, None)
            torch.cuda.synchronize() if DEVICE == "cuda" else None
            timings["crop_torch"].append(time.perf_counter() - t0)

            torch.cuda.synchronize() if DEVICE == "cuda" else None
            t0 = time.perf_counter()
            blur_fov(img_tensor.clone(), gaze, None)
            torch.cuda.synchronize() if DEVICE == "cuda" else None
            timings["blur_torch"].append(time.perf_counter() - t0)

            torch.cuda.synchronize() if DEVICE == "cuda" else None
            t0 = time.perf_counter()
            cm_fov(img_tensor.clone(), gaze, None)
            torch.cuda.synchronize() if DEVICE == "cuda" else None
            timings["cm_torch"].append(time.perf_counter() - t0)

        # -----------------------------
        # Tensor → PIL
        # -----------------------------
        t0 = time.perf_counter()
        img_np = img_tensor.squeeze(0).permute(1,2,0).cpu().numpy()
        img_pil = Image.fromarray(img_np.astype(np.uint8))
        timings["tensor_to_pil"].append(time.perf_counter() - t0)

    # -----------------------------
    # Results
    # -----------------------------

    print("\n===== BENCHMARK RESULTS =====")

    for k, v in timings.items():
        if len(v) > 0:
            print(f"{k:12s}: {np.mean(v)*1000:.2f} ms / image")
            
            
if __name__ == "__main__":
    n_samples = 500
    benchmark_foveations(n_samples)