from pathlib import Path
import json
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image


class OOCDatasetBase(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform

        self.metadata = pd.read_csv(root / "metadata.csv")

        self.class_map = load_imagenet_class_map()

        self.samples = []
        for _, row in self.metadata.iterrows():
            img_name = row["image_id"] + ".JPEG"

            label = self.class_map[row["class_hash"]]
            
            mask_name = img_name.replace(".JPEG", ".png")
            mask_path = self.root / "masks" / mask_name

            mask = np.array(Image.open(mask_path).convert("L"))

            cx, cy = get_gaze_from_mask(mask)

            H, W = mask.shape

            if cx is None:
                cx, cy = W / 2, H / 2  # fallback

            gaze = (cx / W, cy / H)

            self.samples.append({
                "filename": img_name,
                "label": label,
                "gaze": gaze,
                "row": row
            })

    def load_image(self, sample):
        """Override in subclasses"""
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = self.load_image(sample)

        if self.transform:
            img = self.transform(img)

        gaze = sample["gaze"]
        gaze_tensor = torch.tensor(
            [gaze[0], gaze[1]],
            dtype=torch.float32
        )

        return img, sample["label"], gaze_tensor
    
    
class OOCOriginalDataset(OOCDatasetBase):
    def load_image(self, sample):
        path = self.root / "original" / sample["filename"]
        return Image.open(path).convert("RGB")
    

class OOCInpaintedDataset(OOCDatasetBase):
    def load_image(self, sample):
        path = self.root / "inpainted" / sample["filename"]
        return Image.open(path).convert("RGB")
    
    
class OOCObjectOnlyDataset(OOCDatasetBase):
    def load_image(self, sample):
        img_path = self.root / "original" / sample["filename"]
        mask_name = sample["filename"].replace(".JPEG", ".png")
        mask_path = self.root / "masks" / mask_name

        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) > 0

        imagenet_mean = np.array([0.485, 0.456, 0.406]) * 255

        background = np.ones_like(img, dtype=np.float32)

        for c in range(3):
            background[..., c] *= imagenet_mean[c]

        background = background.astype(np.uint8)

        obj_img = np.where(mask[..., None], img, background)

        return Image.fromarray(obj_img.astype(np.uint8))
    
    
class OOCShuffledDataset(OOCDatasetBase):
    def __init__(self, *args, seed=42, **kwargs):
        super().__init__(*args, **kwargs)

        rng = np.random.RandomState(seed)
        self.shuffled_indices = rng.permutation(len(self.samples))
        
    def load_image(self, sample):
        idx = self.samples.index(sample)
        other_idx = self.shuffled_indices[idx]

        other_sample = self.samples[other_idx]

        obj_path = self.root / "original" / sample["filename"]
        mask_name = sample["filename"].replace(".JPEG", ".png")
        mask_path = self.root / "masks" / mask_name
        bg_path = self.root / "inpainted" / other_sample["filename"]

        obj = np.array(Image.open(obj_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) > 0
        bg = np.array(Image.open(bg_path).convert("RGB"))

        # resize bg if needed
        bg = cv2.resize(bg, (obj.shape[1], obj.shape[0]))

        mixed = np.where(mask[..., None], obj, bg)

        return Image.fromarray(mixed.astype(np.uint8))
    
    
def load_imagenet_class_map():
    with open("imagenet_class_index.json") as f:
        data = json.load(f)

    # Format:
    # {"0": ["n01440764", "tench"], ...}

    mapping = {}
    for idx, (synset, _) in data.items():
        mapping[synset] = int(idx)

    return mapping


# helper functions to calculate relative gaze from mask
def get_largest_component(mask):
    """
    mask: binary numpy array (H,W)
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))

    if num_labels <= 1:
        return mask, None  # only background

    # stats: [label, x, y, w, h, area]
    areas = stats[1:, cv2.CC_STAT_AREA]

    largest_idx = 1 + np.argmax(areas)

    largest_mask = (labels == largest_idx).astype(np.uint8)

    return largest_mask


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


def compute_centroid_from_mask(mask):
    m = cv2.moments(mask)
    if m["m00"] == 0:
        return None, None
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return cx, cy


def get_gaze_from_mask(mask):
    mask = (mask > 0).astype(np.uint8)
    mask = fill_mask_holes_floodfill(mask) # like in linear eval
    largest_mask = get_largest_component(mask)
    cx, cy = compute_centroid_from_mask(largest_mask)
    return cx, cy