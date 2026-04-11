
import io
import numpy as np
import pandas as pd
import random
import h5py
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from pathlib import Path
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms.v2 as v2
from torchvision.transforms import PILToTensor
from pycocotools import mask as mask_util
from torchvision.datasets import ImageFolder

from foveation.factory import setup_exact_foveation
from foveation.methods.gaze_crop import GazeCenteredCropGPU
from foveation.methods.radial_blur import RadialBlurFoveation
from foveation.methods.cm import CorticalMagnification

from foveation.ooc.ooc_data import OOCOriginalDataset, OOCInpaintedDataset, OOCObjectOnlyDataset, OOCShuffledDataset
from foveation.crowding import CrowdingDataset, CrowdingDatasetNotMNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    indices = [100_003]

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
    
    viz_fov(samples, method="crop")
    viz_fov(samples, method="blur")
    viz_fov(samples, method="cm")
    # viz_imagenet_mask_samples(4)
    # viz_ooc_datasets(foveation="cm-nosal")
    # viz_crowding_dataset(condition="xax", foveation="blur-light")
    # viz_imagenet_fov_samples(3, remove_padding_bool=True)


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
    
    
def preprocess_like_dataset(img):

    W, H = img.size
    max_side = max(W, H)

    w_ratio = W / max_side
    h_ratio = H / max_side

    # pad bottom/right
    pad_right = max_side - W
    pad_bottom = max_side - H

    img = TF.pad(img, (0,0,pad_right,pad_bottom), fill=0)

    # resize
    img = TF.resize(img, (540,540))

    return img, w_ratio, h_ratio


def remove_padding(img_tensor, ratio):

    B, C, H, W = img_tensor.shape

    valid_W = int(ratio[0] * W)
    valid_H = int(ratio[1] * H)

    return img_tensor[:, :, :valid_H, :valid_W]

    
def viz_imagenet_fov_samples(n, remove_padding_bool):

    crop_fov = GazeCenteredCropGPU()
    blur_fov = RadialBlurFoveation()
    cm_fov = CorticalMagnification()
    val_ds = ImageFolder("/home/data/ILSVRC_real/val", transform=None)

    json_path = "/home/data/elias/imagenet_sam_masks/imagenet_val_gaze_only.json"

    with open(json_path, "r") as f:
        json_data = json.load(f)

    json_by_filename = {
        v["filename"]: v
        for v in json_data.values()
    }

    total = len(val_ds)
    indices = random.sample(range(total), n)

    fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))

    for row, i in enumerate(indices):

        path = val_ds.samples[i][0]
        filename = Path(path).name

        if filename not in json_by_filename:
            print(f"WARNING: {filename} not found in JSON")
            continue

        img = val_ds[i][0]
        dp = json_by_filename[filename]
        
        W_img_original, H_img_original = img.size
        
        img_proc, w_ratio, h_ratio = preprocess_like_dataset(img)
        img_tensor = pil_to_tensor(img_proc).unsqueeze(0)            

        ratio = torch.tensor([w_ratio, h_ratio])
        
    
        # OPTIONAL padding removal
        if remove_padding_bool:
            img_tensor = remove_padding(img_tensor, ratio)

        _, _, H_img, W_img = img_tensor.shape            

        cx_rel = dp["centroid"]["x_rel"]
        cy_rel = dp["centroid"]["y_rel"]

        cx_abs_original = cx_rel * W_img_original
        cy_abs_original = cy_rel * H_img_original
        
        cx_abs = cx_rel * W_img
        cy_abs = cy_rel * H_img

        gaze = torch.tensor([[cx_abs, cy_abs]], dtype=torch.float32)

        with torch.no_grad():

            crop_img = crop_fov(img_tensor.clone(), gaze, None)
            blur_img = blur_fov(img_tensor.clone(), gaze, None)
            cm_img = cm_fov(img_tensor.clone(), gaze, None)

        def to_np(x):
            x = x.squeeze(0).permute(1,2,0).cpu().numpy()
            return x.astype(np.uint8)

        crop_np = to_np(crop_img)
        blur_np = to_np(blur_img)
        cm_np = to_np(cm_img)

        img_np = np.array(img)

        axes[row,0].imshow(img_np)
        axes[row,0].scatter(cx_abs_original, cy_abs_original, c="red", s=20)
        axes[row,0].set_title("Original")

        axes[row,1].imshow(crop_np)
        axes[row,1].set_title("Crop")

        axes[row,2].imshow(blur_np)
        axes[row,2].set_title("Blur")

        axes[row,3].imshow(cm_np)
        axes[row,3].set_title("CM")

        for col in range(4):
            axes[row,col].axis("off")

    plt.tight_layout()
    
    suffix = "nopad" if remove_padding_bool else "withpad"
    save_name = f"imagenet_examples_{suffix}.png"
    
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / "imagenet_fovs" / save_name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {save_name}")
    
    
def viz_imagenet_mask_samples(n):
    
    val_ds = ImageFolder("/home/data/ILSVRC_real/val", transform=None)

    # should't work with this anymore!
    json_path = "/home/data/elias/imagenet_sam_masks/imagenet_val_gaze_only.json"
        
    with open(json_path, "r") as f:
        json_data = json.load(f)
        
    json_by_filename = {
        v["filename"]: v
        for v in json_data.values()
    }

    total = len(val_ds)
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
        
        mask = mask_util.decode(dp["rle"])
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

        title = f"Index {i}\n"
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


def viz_ooc_datasets(n_samples=3, foveation=None):

    root = Path("/home/data/elias/ImageNet-OOC1k_flattened")

    T_pre = v2.Compose([
        v2.Resize(540),
        v2.ToImage(),
        v2.ToDtype(torch.uint8)
    ])

    common_kwargs = dict(
        root=root,
        transform=T_pre,
    )
    
    datasets_dict = {
        "original": OOCOriginalDataset(**common_kwargs), 
        "inpainted": OOCInpaintedDataset(**common_kwargs), 
        "object": OOCObjectOnlyDataset(**common_kwargs), 
        "shuffle": OOCShuffledDataset(**common_kwargs)
    }
    
    if foveation:
        foveation = setup_exact_foveation(foveation)
    
    dataset_names = list(datasets_dict.keys())
    num_datasets = len(dataset_names)

    fig, axes = plt.subplots(
        n_samples,
        num_datasets,
        figsize=(3*num_datasets, 3*n_samples)
    )

    indices = random.sample(range(len(next(iter(datasets_dict.values())))), n_samples)

    for row, idx in enumerate(indices):

        for col, name in enumerate(dataset_names):

            dataset = datasets_dict[name]

            img, label, gaze = dataset[idx]

            if torch.is_tensor(img):
                img_np = img.permute(1,2,0).cpu().numpy()
            else:
                img_np = np.array(img)

            H, W = img_np.shape[:2]

            gx = gaze[0].item() * W
            gy = gaze[1].item() * H

            # optional foveation
            if foveation is not None:
                img_tensor = torch.tensor(img_np).permute(2,0,1).unsqueeze(0).float()
                gaze_abs = torch.tensor([[gx, gy]])

                with torch.no_grad():
                    img_fov = foveation(img_tensor, gaze_abs, None)

                img_np = img_fov.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)

            ax = axes[row, col] if n_samples > 1 else axes[col]

            ax.imshow(img_np)
            ax.scatter(gx, gy, c="red", s=30)

            ax.set_title(f"{name}\nlabel={label}")
            ax.axis("off")

    plt.tight_layout()
    
    save_name=f"ooc_example.png"
    base_dir = Path(__file__).resolve().parent
    out_path = base_dir / "plots" / "ooc" / save_name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {save_name}")
    
    
def viz_crowding_dataset(condition, foveation=None, n=3):
    
    imagenet_val_path = "/home/data/ILSVRC_real/val"
    json_path = "/home/data/elias/imagenet_sam_masks/imagenet_val_masks_with_center.json"
    notmnist_path = "/home/data/elias/notMNIST_small"

    # dataset = CrowdingDataset(imagenet_val_path, json_path, transform=PILToTensor(), condition=condition)
    dataset = CrowdingDatasetNotMNIST(imagenet_val_path, json_path, notmnist_path, transform=PILToTensor(), condition="xax")
    indices = random.sample(range(len(dataset)), n)
    
    if foveation:
        foveation = setup_exact_foveation(foveation)

    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))

    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):

        img, label, gaze = dataset[idx]

        # tensor → numpy
        if torch.is_tensor(img):
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = np.array(img)

        H, W = img_np.shape[:2]

        gx = gaze[0].item() * W
        gy = gaze[1].item() * H

        # optional foveation
        if foveation is not None:
            img_tensor = torch.tensor(img_np).permute(2,0,1).unsqueeze(0).float()
            gaze_abs = torch.tensor([[gx, gy]])

            with torch.no_grad():
                img_fov = foveation(img_tensor, gaze_abs, None)

            img_np = img_fov.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)
                
        ax.imshow(img_np)
        ax.scatter(gx, gy, c="red", s=40)

        ax.set_title(f"Label: {label}")
        ax.axis("off")

    plt.suptitle(f"Crowding Condition: {dataset.condition}")
    plt.tight_layout()
    
    save_name=f"crowding_example_{condition}.png"
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