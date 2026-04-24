import random
import numpy as np
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import cv2
import pandas as pd
import json

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from foveation.utils import ImageNetMaskLoader, load_imagenet_class_map, IMAGENET_VAL_PATH


BASE_CSV_PATH = Path('/home/elias/solo-learn/foveation/analysis/outputs/data/size_analysis/base.csv')
RESULTS_PATH = Path('/home/elias/solo-learn/foveation/analysis/outputs/data/crowding/')
NOTMNIST_PATH = Path('/home/data/elias/notMNIST_small')


def collect_top100_classes():
    df = pd.read_csv(BASE_CSV_PATH)
    name_map = load_imagenet_class_map()
    # aggregate per class
    cls=(
        df.groupby('label')
        .agg(
            accuracy=('correct','mean'),
            mean_mask_area=('mask_area','mean'),
            std_mask_area=('mask_area','std'),
            min_mask_area=('mask_area','min'),
            max_mask_area=('mask_area','max'),
            q05_mask_area=('mask_area', lambda x: x.quantile(0.05)),
            q95_mask_area=('mask_area', lambda x: x.quantile(0.95)),
            n_samples=('label','size'),
            mean_conf=('conf_top1','mean')
        )
        .reset_index()
    )
    cls['accuracy']=cls['accuracy']*100
    # add readable label
    cls['class_name']=cls['label'].map(name_map)
    # sort easiest
    cls=cls.sort_values(by='accuracy', ascending=False)
    top=cls.head(100)
    # column order
    top=top[
        [
        'label',
        'class_name',
        'accuracy',
        'mean_mask_area',
        'std_mask_area',
        'min_mask_area',
        'max_mask_area',
        'q05_mask_area',
        'q95_mask_area',
        'n_samples',
        'mean_conf'
        ]
    ]
    out_path = RESULTS_PATH / "top100_easy_classes.csv"
    top.to_csv(out_path,index=False)
    print(top.head(20))
    print('\nSaved ->',out_path)


def collect_crowding_samples(n_samples=1000, top_k=100, min_area=0.10, max_area=0.30, only_correct=True, min_conf=0.7, per_class_cap=30):
    df = pd.read_csv(BASE_CSV_PATH)
    cls = pd.read_csv(RESULTS_PATH / "top100_easy_classes.csv")
    top_labels = cls.sort_values("accuracy", ascending=False).head(top_k)["label"]
    df = df[df["label"].isin(top_labels)]
    df = df[(df["mask_area"] >= min_area) & (df["mask_area"] <= max_area)]
    if only_correct:
        df = df[df["correct"] == 1]
    if min_conf > 0:
        df = df[df["conf_top1"] >= min_conf]
    # restrict amount of samples per class
    df = df.groupby("label", group_keys=False).apply(lambda x: x.sample(min(len(x), per_class_cap), random_state=0))
    if len(df) > n_samples:
        df = df.sample(n_samples, random_state=0)
    df = df.sort_values("idx").reset_index(drop=True)
    out_path = RESULTS_PATH / "crowding1000.csv"
    df.to_csv(out_path, index=False)
    print(f"{len(df)} samples saved → {out_path}")
    
    
class CrowdingDataset(Dataset):

    def __init__(self,
                 transform=None,
                 condition="xax",          # a, ax, xax
                 canvas_size=540,
                 flanker_distance=20,
                 flanker_size=40):

        self.dataset = ImageFolder(IMAGENET_VAL_PATH)
        self.mask_loader = ImageNetMaskLoader()
        self.flankers = ImageFolder(NOTMNIST_PATH)
        self.transform = transform
        self.condition = condition
        self.canvas_size = canvas_size
        self.flanker_distance = flanker_distance
        self.flanker_size = flanker_size
        
        crop_margin = (canvas_size - (336/540)*canvas_size) // 2
        self.safe_min = crop_margin + flanker_size // 2
        self.safe_max = self.canvas_size - crop_margin - flanker_size // 2
        
        df = pd.read_csv(RESULTS_PATH / "crowding1000.csv")
        self.indices = df["idx"].tolist()

    def __len__(self):
        return len(self.indices)

    def extract_object(self, img, mask):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        obj = img[y0:y1+1, x0:x1+1]
        mask = mask[y0:y1+1, x0:x1+1]
        return obj, mask
    
    def load_flanker(self, target_obj):
        while True:
            try:
                h, w = target_obj.shape[:2]   
                path, _ = random.choice(self.flankers.samples)
                img = Image.open(path).convert("L").resize((self.flanker_size, self.flanker_size))
                arr = np.array(img)
                mask = arr > 30
                obj = np.stack([arr]*3, axis=-1)
                return obj, mask
            except (UnidentifiedImageError, OSError): # some files are corrupt...
                continue
    
    def paste(self, canvas, obj, mask, cx, cy):
        H, W = canvas.shape[:2]
        h, w = obj.shape[:2]
        x0, y0 = int(cx - w//2), int(cy - h//2)
        x1, y1 = x0 + w, y0 + h
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            return canvas
        canvas[y0:y1, x0:x1][mask] = obj[mask]
        return canvas

    def __getitem__(self, i):
        # --- load image ---
        idx = self.indices[i]
        path, label = self.dataset.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        filename = Path(path).name
        mask = self.mask_loader.get_mask(filename)
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        out = self.extract_object(img, mask)
        if out is None:
            return self.__getitem__((idx+1) % len(self))
        obj, mask = out
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        cx = cy = self.canvas_size // 2
        canvas = self.paste(canvas, obj, mask, cx, cy)
        
        h, w = obj.shape[:2]
        obj_left  = cx - w // 2
        obj_right = cx + w // 2
        
        d = self.flanker_distance  
        
        if self.condition == "ax":
            f_obj, f_mask = self.load_flanker(obj)
            fh, fw = f_obj.shape[:2]
            flanker_cx = np.clip(obj_right + d + fw // 2, self.safe_min, self.safe_max)
            canvas = self.paste(canvas, f_obj, f_mask, flanker_cx, cy)
        elif self.condition == "xax":
            f_obj, f_mask = self.load_flanker(obj)
            fh, fw = f_obj.shape[:2]
            flanker_cx_left = np.clip(obj_left  - d - fw // 2, self.safe_min, self.safe_max)
            flanker_cx_right = np.clip(obj_right + d + fw // 2, self.safe_min, self.safe_max)
            canvas = self.paste(canvas, f_obj, f_mask, flanker_cx_left, cy)
            canvas = self.paste(canvas, f_obj, f_mask, flanker_cx_right, cy)
        # --- output ---
        img_out = Image.fromarray(canvas)
        if self.transform:
            img_out = self.transform(img_out)
        if self.condition == "xa":
            gx = (cx + d) / self.canvas_size
            gy = cy / self.canvas_size
        else:
            gx = cx / self.canvas_size
            gy = cy / self.canvas_size
        gaze = torch.tensor([gx, gy], dtype=torch.float32)
        return img_out, label, gaze



if __name__ == "__main__":
    print("-----------------------------------------------------------------------------------------------------------------------")
    #collect_top100_classes()
    #collect_crowding_samples()