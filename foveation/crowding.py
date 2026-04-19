import random
import csv
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import PILToTensor
import torchvision.transforms.v2 as v2

from foveation.factory import setup_exact_foveation
from foveation.utils import ImageNetMaskLoader
from foveation.ooc.ooc_utils import load_model, IdentityFoveation
    
    
class CrowdingDatasetNotMNIST(Dataset):
    def __init__(
        self,
        imagenet_root,
        mask_json,
        notmnist_root,
        transform=None,
        condition="a",   # a, xa, ax, xax
        canvas_size=540,
        object_size=120,
        flanker_size=60
    ):
        # --- target dataset ---
        self.dataset = ImageFolder(root=imagenet_root)
        self.mask_loader = ImageNetMaskLoader(json_path=mask_json, fill_holes=True)

        # --- flanker dataset ---
        self.notmnist = ImageFolder(root=notmnist_root)

        self.transform = transform
        self.condition = condition

        self.canvas_size = canvas_size
        self.object_size = object_size
        self.flanker_size = flanker_size

    def __len__(self):
        return len(self.dataset)

    # -----------------------
    # TARGET HELPERS
    # -----------------------

    def extract_object(self, img, mask):
        mask = (mask > 0).astype(np.uint8)

        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        obj = img[y0:y1+1, x0:x1+1]
        mask = mask[y0:y1+1, x0:x1+1]

        return obj, mask

    def resize_object(self, obj, mask):

        h, w = obj.shape[:2]
        if h == 0 or w == 0:
            return None
        
        scale = self.object_size / max(h, w)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        obj = Image.fromarray(obj).resize((new_w, new_h), Image.BILINEAR)
        mask = Image.fromarray(mask.astype(np.uint8)*255).resize((new_w, new_h), Image.NEAREST)

        return np.array(obj), (np.array(mask) > 0)

    # -----------------------
    # FLANKER HELPERS
    # -----------------------

    def load_flanker(self):
        while True:
            try:
                idx = random.randint(0, len(self.notmnist)-1)
                path, _ = self.notmnist.samples[idx]

                img = Image.open(path).convert("L")
                img = img.resize(
                    (self.flanker_size, self.flanker_size),
                    Image.BILINEAR
                )

                img_np = np.array(img)
                mask = img_np > 30
                obj = np.stack([img_np]*3, axis=-1)
                return obj, mask

            except (UnidentifiedImageError, OSError):
                # corrupted image → sample another flanker
                continue

    # -----------------------
    # CANVAS
    # -----------------------

    def create_background(self):
        H = W = self.canvas_size
        return np.zeros((H, W, 3), dtype=np.uint8)

    def paste(self, canvas, obj, mask, cx, cy):
        H, W = canvas.shape[:2]
        h, w = obj.shape[:2]

        x0 = int(cx - w // 2)
        y0 = int(cy - h // 2)

        x1 = x0 + w
        y1 = y0 + h

        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            return canvas

        region = canvas[y0:y1, x0:x1]
        region[mask] = obj[mask]
        canvas[y0:y1, x0:x1] = region

        return canvas

    # -----------------------
    # MAIN
    # -----------------------

    def __getitem__(self, idx):

        # --- TARGET ---
        path, label = self.dataset.samples[idx]
        filename = Path(path).name

        img = np.array(Image.open(path).convert("RGB"))

        mask = self.mask_loader.get_mask(filename)

        # resize mask
        H_img, W_img = img.shape[:2]
        if mask.shape != (H_img, W_img):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (W_img, H_img),
                interpolation=cv2.INTER_NEAREST
            )

        obj = self.extract_object(img, mask)
        if obj is None:
            return self.__getitem__((idx + 1) % len(self))

        obj, mask = obj
        obj, mask = self.resize_object(obj, mask)
        
        obj_f, mask_f = self.load_flanker()

        # --- CANVAS ---
        canvas = self.create_background()
        H = W = self.canvas_size

        cx_target = int(W * 0.5)
        cy_target = int(H * 0.5)

        # --- FLANKERS ---
        if self.condition == "xa":
            cx_target = int(W * 0.75)
            canvas = self.paste(canvas, obj_f, mask_f, int(W * 0.5), cy_target)
        elif self.condition == "ax":  
            cx_target = int(W * 0.5)
            canvas = self.paste(canvas, obj_f, mask_f, int(W * 0.75), cy_target)
        elif self.condition == "xax":
            cx_target = int(W * 0.5)
            canvas = self.paste(canvas, obj_f, mask_f, int(W * 0.25), cy_target)
            canvas = self.paste(canvas, obj_f, mask_f, int(W * 0.75), cy_target)
        else:
            pass

        canvas = self.paste(canvas, obj, mask, cx_target, cy_target)
        
        # --- TO TENSOR ---
        img_out = Image.fromarray(canvas)

        if self.transform:
            img_out = self.transform(img_out)

        gaze = torch.tensor([
            cx_target / W,
            cy_target / H
        ], dtype=torch.float32)

        return img_out, label, gaze
    
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def evaluate_crowding(model, device, model_name, foveation, seed, imagenet_root, json_path, notmnist_root):

    set_seed(seed)
    model.eval()
    model.to(device)
    conditions = ["a", "xa", "ax", "xax"]

    stats = {c: {"correct": 0, "conf_sum": 0.0, "n": 0} for c in conditions}

    # --- post-transform ---
    T_post = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # --- evaluate each condition ---
    for cond in conditions:

        print(f"\nCondition: {cond}")

        ds = CrowdingDatasetNotMNIST(
            imagenet_root=imagenet_root, 
            mask_json=json_path, 
            notmnist_root=notmnist_root, 
            transform=PILToTensor(), 
            condition=cond)

        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

        with torch.no_grad():

            pbar = tqdm(loader, total=len(loader))

            for img, label, gaze in pbar:
                img = img.to(device)
                label = label.to(device)
                gaze = gaze.to(device)

                # convert relative gaze to absolute gaze
                B, C, H, W = img.shape
                gaze_abs = gaze.clone()
                gaze_abs[0, 0] *= W
                gaze_abs[0, 1] *= H 

                img = foveation(img, gaze_abs)
                img = T_post(img)

                logits = model(img)

                probs = torch.softmax(logits, dim=1)
                top5_probs, top5_idx = torch.topk(probs, k=5, dim=1)
                pred = top5_idx[:,0]
                correct = (pred == label)

                stats[cond]["correct"] += int(correct.item())
                # top1 confidence
                stats[cond]["conf_sum"] += top5_probs[0,0].item()
                stats[cond]["n"] += 1

    # summarize
    result = {"foveation": model_name, "seed": seed}

    for cond in conditions:
        acc = stats[cond]["correct"] / stats[cond]["n"]
        conf = stats[cond]["conf_sum"] / stats[cond]["n"]
        result[f"{cond}_acc"] = acc
        result[f"{cond}_conf"] = conf

    return result


def save_result(result, out_path):
    write_header = not Path(out_path).exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)
    print(f"Saved -> {out_path}")
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    
    imagenet_val_path = "/home/data/ILSVRC_real/val"
    json_path = "/home/data/elias/imagenet_sam_masks/imagenet_val_masks_with_center.json"
    notmnist_path = "/home/data/elias/notMNIST_small"
    
    results_path = Path("/home/elias/solo-learn/foveation/analysis/outputs/data/crowding_results.csv")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # model
    model = load_model(args.model).to(device)

    # foveation
    if args.model in ["base", "dummy"]:
        foveation = IdentityFoveation()
    else:
        foveation = setup_exact_foveation(args.model)
        
    foveation = foveation.to(device)
    
    result = evaluate_crowding(
        model=model,
        device=device,
        model_name=args.model,
        foveation=foveation,
        seed=args.seed,
        imagenet_root=imagenet_val_path,
        json_path=json_path,
        notmnist_root=notmnist_path
    )

    save_result(result, results_path)
    