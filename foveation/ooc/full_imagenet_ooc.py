from pathlib import Path
import argparse
import json
import csv
import torch
import random
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from PIL import Image
from pycocotools import mask as mask_utils

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import PILToTensor

from foveation.utils import ImageNetMaskLoader, build_model_and_foveation, T_POST, IMAGENET_VAL_PATH, MASK_JSON_PATH
from foveation.mask_centroids import fill_mask_holes_floodfill

INPAINTED_PATH = Path("/home/data/elias/ImageNet-OOC1k_flattened/inpainted")
RESULTS_PATH = Path("/home/elias/solo-learn/foveation/analysis/outputs/data/full_ooc_results.csv")
BG_OUT_DIR = Path("/home/elias/solo-learn/foveation/analysis/outputs/data/background_area")
BG_OUT_DIR.mkdir(parents=True,exist_ok=True)
    
    
class ImageNetValOOC(Dataset):

    def __init__(self, mode="original", seed=42, transform=PILToTensor()): # original, object, ooc

        self.mode = mode
        self.imagenet = ImageFolder(root=IMAGENET_VAL_PATH)
        self.mask_loader = ImageNetMaskLoader()
        self.transform = transform 

        # --- load inpainted pool ---
        self.inpainted_files = sorted(list(INPAINTED_PATH.glob("*.JPEG")))
        # random background assignment
        rng = np.random.RandomState(seed)
        self.bg_indices = rng.randint(0, len(self.inpainted_files), size=len(self.imagenet.samples))

    def __len__(self):
        return len(self.imagenet)

    def create_background(self, img):
        imagenet_mean = np.array([0.485, 0.456, 0.406]) * 255
        bg = np.ones_like(img, dtype=np.float32)
        for c in range(3):
            bg[...,c] *= imagenet_mean[c]
        return bg.astype(np.uint8)

    def load_image(self, idx):
        path, _ = self.imagenet.samples[idx]
        filename = Path(path).name
        img = np.array(Image.open(path).convert("RGB"))
        if self.mode == "original":
            return Image.fromarray(img)
        # load foreground mask
        mask = self.mask_loader.get_mask(filename)
        if mask is None:
            return Image.fromarray(img)
        # safety if dimensions mismatch
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        # object-only
        if self.mode == "object":
            bg = self.create_background(img)
            out = np.where(mask[...,None], img, bg)
            return Image.fromarray(out.astype(np.uint8))
        # shuffled OOC
        elif self.mode == "ooc":
            bg_path = self.inpainted_files[self.bg_indices[idx]]
            bg = np.array(Image.open(bg_path).convert("RGB"))
            bg = cv2.resize(bg,(img.shape[1], img.shape[0]))
            out = np.where(mask[...,None], img, bg)
            return Image.fromarray(out.astype(np.uint8))
        else:
            raise ValueError(self.mode)

    def __getitem__(self, idx):
        path, label = self.imagenet.samples[idx]
        filename = Path(path).name
        img = self.load_image(idx)
        if self.transform:
            img = self.transform(img)
        gaze = self.mask_loader.get_centroid(filename)
        gaze_tensor = torch.tensor(gaze, dtype=torch.float32)
        return img, label, gaze_tensor
    

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def evaluate_full_ooc(model, device, model_name, foveation):
    model.eval()
    model.to(device)

    def evaluate_mode(mode, seed=None):

        if seed is not None:
            set_seed(seed)

        ds = ImageNetValOOC(mode=mode, seed=seed, transform=PILToTensor())
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
        
        correct = 0
        n = 0
        conf_correct_sum = 0.0
        conf_incorrect_sum = 0.0
        n_correct = 0
        n_incorrect = 0

        with torch.no_grad():
            pbar = tqdm(loader, desc=f"{mode} seed={seed}")
            for img, label, gaze in pbar:
                img = img.to(device)
                label = label.to(device)
                gaze = gaze.to(device)

                B,C,H,W = img.shape
                gaze_abs = gaze.clone()
                gaze_abs[:,0] *= W
                gaze_abs[:,1] *= H

                img = foveation(img, gaze_abs)
                img = T_POST(img)

                logits = model(img)
                probs = torch.softmax(logits, dim=1)

                pred = probs.argmax(dim=1)
                conf = probs.max().item()
                
                if pred == label:
                    correct += 1
                    conf_correct_sum += conf
                    n_correct += 1
                else:
                    conf_incorrect_sum += conf
                    n_incorrect += 1
                n += 1

        return {
            "acc": correct / n,
            "conf_correct": conf_correct_sum / max(n_correct,1),
            "conf_incorrect": conf_incorrect_sum / max(n_incorrect,1)
        }

    print("\nEvaluating ORIGINAL")
    original = evaluate_mode("original")

    print("\nEvaluating OBJECT")
    object_res = evaluate_mode("object")

    # OOC over seeds
    ooc_runs = []
    for seed in SEEDS:
        print(f"\nEvaluating OOC seed={seed}")
        res = evaluate_mode("ooc", seed=seed)
        ooc_runs.append(res)
        
    ooc_accs = [r["acc"] for r in ooc_runs]
    ooc_conf_corrects = [r["conf_correct"] for r in ooc_runs]
    ooc_conf_incorrects = [r["conf_incorrect"] for r in ooc_runs]
    ooc_acc = np.mean(ooc_accs)
    ooc_acc_std = np.std(ooc_accs)
    ooc_conf_correct = np.mean(ooc_conf_corrects)
    ooc_conf_incorrect = np.mean(ooc_conf_incorrects)

    return {
        "foveation": model_name,

        "original_acc": original["acc"],
        "original_conf_correct": original["conf_correct"],
        "original_conf_incorrect": original["conf_incorrect"],
        
        "object_acc": object_res["acc"],
        "object_conf_correct": object_res["conf_correct"],
        "object_conf_incorrect": object_res["conf_incorrect"],
        
        "ooc_acc": ooc_acc,
        "ooc_acc_std": ooc_acc_std,
        "ooc_conf_correct": ooc_conf_correct,
        "ooc_conf_incorrect": ooc_conf_incorrect
    }


def save_result(result):
    write_header = not Path(RESULTS_PATH).exists()
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)
    print(f"Saved -> {RESULTS_PATH}")
    
    
def calculate_foveated_mask_areas(foveation, model_name):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    foveation = foveation.to(device)
    foveation.eval()

    with open(MASK_JSON_PATH) as f:
        mask_data = json.load(f)

    results = []
    for entry in tqdm(mask_data.values(), desc=model_name):

        mask = mask_utils.decode(entry["rle"])
        mask = fill_mask_holes_floodfill(mask)
        h,w = mask.shape

        area_orig = (mask.sum() / (h*w))
        mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        gaze = torch.tensor([[
            entry["centroid"]["x_rel"] * w,
            entry["centroid"]["y_rel"] * h
        ]],dtype=torch.float32).to(device)

        with torch.no_grad():
            fov_mask = foveation(mask_t, gaze)
        fov_mask = (fov_mask > 0).float()
        _, _, h_fov, w_fov = fov_mask.shape
        area_fov = fov_mask.sum().item() / (h_fov * w_fov)
        # safety checks
        if area_orig <= 0:
            continue
        if area_fov <= 0:
            area_fov = 0.0
        area_ratio = (area_fov / area_orig)
        area_delta = (area_fov - area_orig)

        results.append({
            "orig_area": area_orig,
            "fov_area": area_fov,
            "area_ratio": area_ratio,
            "area_delta": area_delta
        })

    df = pd.DataFrame(results)
    out_path = (BG_OUT_DIR / f"{model_name}.csv")
    df.to_csv(out_path, index=False)

    print(f"\nSaved -> {out_path}")
    print(
        f"{model_name}: "
        f"ratio={df.area_ratio.mean():.3f} +/- {df.area_ratio.std():.3f} | "
        f"fov_area={df.fov_area.mean():.3f} | "
        f"background_retained={(1 - df.fov_area.mean()):.3f}"
    )
    return df
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    
    SEEDS = [1,2,3]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.background:
        for m in ["crop", "cm-nosal", "cm-strong"]:
            _, fov = build_model_and_foveation(m, device)
            fov = fov.to(device)
            calculate_foveated_mask_areas(fov, m)
    else:
        model, foveation = build_model_and_foveation(args.model, device)
        result = evaluate_full_ooc(model, device, args.model, foveation)
        save_result(result)
