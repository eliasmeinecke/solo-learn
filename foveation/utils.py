from pathlib import Path
import json
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
