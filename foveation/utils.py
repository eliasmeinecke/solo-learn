from pathlib import Path
import json
import random
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import cv2
import numpy as np
from pycocotools import mask as mask_utils
    

class ImageNetSizeDataset(Dataset):
    def __init__(self, root, gaze_json, transform=None):

        self.dataset = ImageFolder(root=root)
        self.transform = transform

        with open(gaze_json, "r") as f:
            gaze_data = json.load(f)

        self.mask_by_filename = {v["filename"]: v for v in gaze_data.values()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        path, label = self.dataset.samples[idx]
        filename = Path(path).name

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # --- gaze + area ---
        dp = self.mask_by_filename.get(filename, None)
        if dp is not None:
            gaze_rel = torch.tensor([
                    dp["centroid"]["x_rel"],
                    dp["centroid"]["y_rel"]
                ], dtype=torch.float32)
            area = torch.tensor(dp["area"], dtype=torch.float32)
        else:
            gaze_rel = torch.tensor([0.5, 0.5], dtype=torch.float32)
            area = torch.tensor(0.0, dtype=torch.float32)

        return img, label, gaze_rel, area
    

def fill_mask_holes_floodfill(mask):
    # https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    mask_padded = cv2.copyMakeBorder(mask_uint8, 1,1,1,1, cv2.BORDER_CONSTANT, value=0)

    h, w = mask_padded.shape
    ff_mask = np.zeros((h+2, w+2), np.uint8)

    cv2.floodFill(mask_padded, ff_mask, (0,0), 255)
    mask_inv = cv2.bitwise_not(mask_padded)

    mask_filled = mask_uint8 | mask_inv[1:-1, 1:-1]
    return (mask_filled > 0).astype(np.uint8)


class ImageNetMaskLoader:
    def __init__(self, json_path, fill_holes=True):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.masks_by_filename = {
            v["filename"]: v for v in data.values()
        }

        self.fill_holes = fill_holes

    def get_mask(self, filename):
        entry = self.masks_by_filename.get(filename, None)

        if entry is None:
            return None

        # --- decode RLE ---
        mask = mask_utils.decode(entry["rle"])  # (H, W)

        # --- optional cleanup ---
        if self.fill_holes:
            mask = fill_mask_holes_floodfill(mask)

        return mask.astype(np.uint8)

    def get_area(self, filename):
        entry = self.masks_by_filename.get(filename, None)
        if entry is None:
            return 0.0
        return entry.get("area_mask_rel", 0.0)

    def get_centroid(self, filename):
        entry = self.masks_by_filename.get(filename, None)
        if entry is None:
            return None
        return (
            entry["centroid"]["x_rel"],
            entry["centroid"]["y_rel"]
        )


class CrowdingDataset(Dataset):
    def __init__(
        self,
        root,
        json_path,
        transform=None,
        condition="a",   # a, xa, ax, xax
        canvas_size=540,
        object_size=120,
        background="gray"
    ):
        self.dataset = ImageFolder(root=root)
        self.transform = transform
        self.condition = condition

        self.canvas_size = canvas_size
        self.object_size = object_size
        self.background = background

        self.mask_loader = ImageNetMaskLoader(json_path=json_path, fill_holes=True)

        # --- group by class (for flanker sampling) ---
        self.class_to_indices = {}
        for i, (_, label) in enumerate(self.dataset.samples):
            self.class_to_indices.setdefault(label, []).append(i)

    def __len__(self):
        return len(self.dataset)

    # -----------------------
    # helpers
    # -----------------------

    def extract_object(self, img, mask):
        """Apply mask and crop object"""
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

        scale = self.object_size / max(h, w)

        new_w = int(w * scale)
        new_h = int(h * scale)

        obj_pil = Image.fromarray(obj)
        mask_pil = Image.fromarray(mask * 255)

        obj_pil = obj_pil.resize((new_w, new_h), Image.BILINEAR)
        mask_pil = mask_pil.resize((new_w, new_h), Image.NEAREST)

        return np.array(obj_pil), (np.array(mask_pil) > 0)

    def create_background(self):
        H = W = self.canvas_size

        if self.background == "gray":
            return np.ones((H, W, 3), dtype=np.uint8) * 127
        elif self.background == "black":
            return np.zeros((H, W, 3), dtype=np.uint8)
        else:
            imagenet_mean = np.array([0.485, 0.456, 0.406]) * 255
            background = np.ones((H, W, 3), dtype=np.uint8)
            for c in range(3):
                background[..., c] *= imagenet_mean[c]
            return background

    def paste(self, canvas, obj, mask, cx, cy):
        H, W = canvas.shape[:2]
        h, w = obj.shape[:2]

        x0 = int(cx - w // 2)
        y0 = int(cy - h // 2)

        x1 = x0 + w
        y1 = y0 + h

        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            print("Skipped paste (out of bounds)")
            return canvas  # skip if out of bounds

        region = canvas[y0:y1, x0:x1]
        region[mask] = obj[mask]
        canvas[y0:y1, x0:x1] = region

        return canvas

    def sample_flanker(self, target_label):
        while True:
            idx = random.randint(0, len(self.dataset) - 1)
            _, label = self.dataset.samples[idx]
            if label != target_label:
                return idx

    # -----------------------
    # main
    # -----------------------

    def __getitem__(self, idx):

        path, label = self.dataset.samples[idx]
        filename = Path(path).name

        img = np.array(Image.open(path).convert("RGB"))

        mask = self.mask_loader.get_mask(filename)
        # important fix: resize mask to image size
        H_img, W_img = img.shape[:2]
        H_mask, W_mask = mask.shape

        if (H_mask != H_img) or (W_mask != W_img):
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

        # --- canvas ---
        canvas = self.create_background()

        H = W = self.canvas_size
        cx_target = int(W * 0.6)
        cy_target = int(H * 0.5)

        # --- paste target ---
        canvas = self.paste(canvas, obj, mask, cx_target, cy_target)

        # --- flankers ---
        if self.condition in ["xa", "xax"]:
            idx_f = self.sample_flanker(label)
            obj_f, mask_f = self._load_object(idx_f)

            cx_center = int(W * 0.35)
            canvas = self.paste(canvas, obj_f, mask_f, cx_center, cy_target)

        if self.condition in ["ax", "xax"]:
            idx_f = self.sample_flanker(label)
            obj_f, mask_f = self._load_object(idx_f)

            cx_periph = int(W * 0.85)
            canvas = self.paste(canvas, obj_f, mask_f, cx_periph, cy_target)

        # --- to PIL ---
        img_out = Image.fromarray(canvas)

        if self.transform:
            img_out = self.transform(img_out)

        # --- gaze (TARGET!) ---
        gaze = torch.tensor([
            cx_target / W,
            cy_target / H
        ], dtype=torch.float32)

        return img_out, label, gaze

    # helper for flankers
    def _load_object(self, idx):
        path, label = self.dataset.samples[idx]
        filename = Path(path).name

        img = np.array(Image.open(path).convert("RGB"))
        mask = self.mask_loader.get_mask(filename)
        # important fix: resize mask to image size
        H_img, W_img = img.shape[:2]
        H_mask, W_mask = mask.shape

        if (H_mask != H_img) or (W_mask != W_img):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (W_img, H_img),
                interpolation=cv2.INTER_NEAREST
            )

        obj = self.extract_object(img, mask)
        obj, mask = obj
        return self.resize_object(obj, mask)