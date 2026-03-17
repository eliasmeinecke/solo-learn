import io
import json
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import PILToTensor

from solo.data.custom.imagenet import ImgNetDataset_42
from foveation.factory import setup_exact_foveation, VALID_FOVS


IMAGENET_PATH = Path("/pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/ImageNet/h5")

# ===============================================
# VERY IMPORTANT: set JSON & OUT_PATH correctly!
# ===============================================
OUT_PATH = Path("/tims/output/path")
JSON_PATH = Path("/tims/json/path/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_foveated_imagenet(foveation, split, out_dir_suffix, max_images=None):

    out_dir = OUT_PATH / out_dir_suffix / "ImageNet" / "h5"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{split}.h5"
    
    json_path = JSON_PATH / f"imagenet_{split}_masks_with_center.json"
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    if out_path.exists():
        print(f"[Dataset] {out_path} already exists.")
        return str(OUT_PATH / out_dir_suffix)

    print(f"[Dataset] Building {out_path}")
    
    dataset = ImgNetDataset_42(
        IMAGENET_PATH,
        transform=PILToTensor(),
        split=split,
        subset=None,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,      
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    dataset_size = len(dataset)
    
    if max_images is not None:
        dataset_size = min(dataset_size, max_images)

    dtype = h5py.vlen_dtype(np.dtype("uint8"))

    with open(json_path) as f:
        gaze_data = json.load(f)

    mask_by_filename = {v["filename"]: v for v in gaze_data.values()}
    filenames = dataset.mapper["filename"].tolist()
    
    foveation = foveation.to(device)
    
    with h5py.File(out_path, "w") as out_h5:

        out_images = out_h5.create_dataset("images", shape=(dataset_size,), dtype=dtype)
        out_targets = out_h5.create_dataset("targets", shape=(dataset_size,), dtype=np.int32)

        pbar = tqdm(dataloader, desc=f"[{out_dir_suffix}] {split}", total=dataset_size)
        
        for i, data in enumerate(pbar):
            
            if i >= dataset_size:
                break
            
            if i % 100 == 0 and i > 0:
                pbar.set_postfix({"img/s": f"{i / max(pbar.format_dict['elapsed'], 1e-6):.1f}"})

            img, target = data
            img = img.to(device, non_blocking=True)
            
            filename = filenames[i]
            dp = mask_by_filename.get(filename, None)

            if dp is not None:
                gaze_rel = torch.tensor([
                    dp["centroid"]["x_rel"],
                    dp["centroid"]["y_rel"]
                ], device=device, dtype=torch.float32)
            else:
                gaze_rel = torch.tensor([0.5, 0.5], device=device, dtype=torch.float32)

            # compute absolute gaze
            _, _, H, W = img.shape
            gaze_abs = torch.tensor([
                gaze_rel[0] * W,
                gaze_rel[1] * H
            ], device=device).view(1,2)

            with torch.no_grad():
                img_fov = foveation(img, gaze_abs, None)

            img_fov = img_fov.squeeze(0).permute(1,2,0).cpu().numpy()

            img_pil = Image.fromarray(img_fov.astype(np.uint8))

            # encode JPEG
            buf = io.BytesIO()
            img_pil.save(buf, format="JPEG", quality=90)

            out_images[i] = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            out_targets[i] = target.item()
        
    print(f"[Dataset] Using foveated ImageNet: {out_dir_suffix}")
    return str(OUT_PATH / out_dir_suffix)


if __name__ == "__main__":
    # ===============================================
    # example cmds:
    # python -m foveation.foveated_imagenet --fov blur-light
    # python -m foveation.foveated_imagenet --fov crop ----max-images 50
    # python -m foveation.foveated_imagenet --fov cm-strong ----max-images 1000
    # ===============================================
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fov", type=str, required=True)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    build_fov_type = args.fov
    max_images = args.max_images
    
    suffix = build_fov_type
    if max_images is not None:
        suffix += f"-debug{max_images}"
    
    if build_fov_type in VALID_FOVS:
        foveation = setup_exact_foveation(build_fov_type)
        build_foveated_imagenet(foveation, split="train", out_dir_suffix=suffix, max_images=max_images)
        build_foveated_imagenet(foveation, split="val", out_dir_suffix=suffix, max_images=max_images)
    else:
        print(f"No valid foveation type: {build_fov_type}. NOT building foveated imagenet version.")
    