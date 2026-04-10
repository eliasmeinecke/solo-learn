import random
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import PILToTensor
from foveation.utils import ImageNetMaskLoader


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
        flanker_size=80,
        background="black"
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
        self.background = background

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
        scale = self.object_size / max(h, w)

        new_w = int(w * scale)
        new_h = int(h * scale)

        obj = Image.fromarray(obj).resize((new_w, new_h), Image.BILINEAR)
        mask = Image.fromarray(mask * 255).resize((new_w, new_h), Image.NEAREST)

        return np.array(obj), (np.array(mask) > 0)

    # -----------------------
    # FLANKER HELPERS
    # -----------------------

    def load_flanker(self):
        idx = random.randint(0, len(self.notmnist) - 1)
        path, _ = self.notmnist.samples[idx]

        img = Image.open(path).convert("L")  # grayscale
        img = img.resize((self.flanker_size, self.flanker_size), Image.BILINEAR)

        img_np = np.array(img)

        # normalize to binary-ish mask
        mask = img_np > 30

        # convert to RGB
        obj = np.stack([img_np]*3, axis=-1)

        return obj, mask

    # -----------------------
    # CANVAS
    # -----------------------

    def create_background(self):
        H = W = self.canvas_size

        if self.background == "gray":
            return np.ones((H, W, 3), dtype=np.uint8) * 127
        elif self.background == "black":
            return np.zeros((H, W, 3), dtype=np.uint8)
        else:
            imagenet_mean = np.array([0.485, 0.456, 0.406]) * 255
            bg = np.ones((H, W, 3), dtype=np.uint8)
            for c in range(3):
                bg[..., c] *= imagenet_mean[c]
            return bg

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

        # --- CANVAS ---
        canvas = self.create_background()
        H = W = self.canvas_size

        cx_target = int(W * 0.6)
        cy_target = int(H * 0.5)

        canvas = self.paste(canvas, obj, mask, cx_target, cy_target)

        # --- FLANKERS ---
        if self.condition in ["xa", "xax"]:
            obj_f, mask_f = self.load_flanker()
            canvas = self.paste(canvas, obj_f, mask_f, int(W * 0.35), cy_target)

        if self.condition in ["ax", "xax"]:
            obj_f, mask_f = self.load_flanker()
            canvas = self.paste(canvas, obj_f, mask_f, int(W * 0.85), cy_target)

        # --- TO TENSOR ---
        img_out = Image.fromarray(canvas)

        if self.transform:
            img_out = self.transform(img_out)

        gaze = torch.tensor([
            cx_target / W,
            cy_target / H
        ], dtype=torch.float32)

        return img_out, label, gaze
    
    
    
if __name__ == "__main__":
    
    imagenet_val_path = "/home/data/ILSVRC_real/val"
    json_path = "/home/data/elias/imagenet_sam_masks/imagenet_val_masks_with_center.json"
    notmnist_path = "/home/data/elias/notMNIST_small"
    
    dataset = CrowdingDataset(
        root=imagenet_val_path,
        json_path=json_path,
        transform=PILToTensor(),
        condition="xax"
    )
    
    dataset = CrowdingDatasetNotMNIST(
        imagenet_root=imagenet_val_path,
        json_path=json_path,
        notmnist_root=notmnist_path,
        transform=PILToTensor(),
        condition="xax"
    )
    
    