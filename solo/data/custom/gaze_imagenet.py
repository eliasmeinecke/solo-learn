import io
import json
import os
from pathlib import Path
from typing import Union, Callable, Optional, Tuple
import cv2
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as TF

from solo.data.custom.imagenet import ImgNetDataset_42


class ImgNetWithGaze(Dataset):

    def __init__(self, root, transform, split):
        
        self.base_dataset = ImageFolder(root)
        
        self.filter_ooc_images()

        json_path = f"/home/data/elias/imagenet_sam_masks/imagenet_{split}_masks_with_center.json"

        with open(json_path, "r") as f:
            data = json.load(f)

        self.mask_by_filename = {
            v["filename"]: v
            for v in data.values()
        }
        
        self.transform = transform # make sure the transform is set correctly!
        
    def __len__(self):
        return len(self.base_dataset)

    def filter_ooc_images(self):

        ooc_txt_path = "/home/elias/solo-learn/solo/data/dataset_subset/ooc_image_ids.txt"
        
        with open(ooc_txt_path, "r") as f:
            excluded = set(line.strip() for line in f)

        filtered_samples = []

        for path, label in self.base_dataset.samples:
            filename = os.path.basename(path)

            if filename not in excluded:
                filtered_samples.append((path, label))

        self.base_dataset.samples = filtered_samples
        self.base_dataset.imgs = filtered_samples
        self.base_dataset.targets = [s[1] for s in filtered_samples]
    
    
    def pad_to_square(self, img):
        w, h = img.size
        max_side = max(w, h)

        pad_right = max_side - w
        pad_bottom = max_side - h

        img = TF.pad(img, (0, 0, pad_right, pad_bottom), fill=0)

        return img

    def __getitem__(self, idx):

        img, label = self.base_dataset[idx]

        path, _ = self.base_dataset.samples[idx]
        filename = os.path.basename(path)

        dp = self.mask_by_filename.get(filename, None)

        if dp is not None:
            gaze_rel = torch.tensor([dp["centroid"]["x_rel"], dp["centroid"]["y_rel"]], dtype=torch.float32)
            area_rel = torch.tensor(dp.get("area_mask_rel", 0.0), dtype=torch.float32)
        else:
            gaze_rel = torch.tensor([0.5, 0.5], dtype=torch.float32)
            area_rel = torch.tensor(0.0, dtype=torch.float32)

        W_original, H_original = img.size

        max_side = max(W_original, H_original)

        w_ratio = W_original / max_side
        h_ratio = H_original / max_side

        ratio = torch.tensor([w_ratio, h_ratio], dtype=torch.float32)

        img = self.pad_to_square(img)
        
        if self.transform is not None:
            img = self.transform(img)

        meta = {
            "gaze": gaze_rel,
            "ratio": ratio,
            "area": area_rel
        }

        return img, label, meta

    
    
class ImgNetDataset_42_Gaze(ImgNetDataset_42):

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        split: str = "train",
        subset: Optional[str] = None,
    ):

        super().__init__(root, transform=None, split=split, subset=subset)

        self.transform = transform

        json_path = f"/home/data/elias/imagenet_sam_masks/imagenet_{split}_masks_with_center.json"

        with open(json_path, "r") as f:
            data = json.load(f)

        self.mask_by_filename = {
            v["filename"]: v
            for v in data.values()
        }

        self.filter_ooc_images()

    def filter_ooc_images(self):

        ooc_txt_path = "/home/elias/solo-learn/solo/data/dataset_subset/ooc_image_ids.txt"

        with open(ooc_txt_path, "r") as f:
            excluded = set(line.strip() for line in f)

        if self.mapper is not None:
            self.mapper = self.mapper[~self.mapper["filename"].isin(excluded)].reset_index(drop=True)

    def pad_to_square(self, img):

        w, h = img.size
        max_side = max(w, h)

        pad_right = max_side - w
        pad_bottom = max_side - h

        img = TF.pad(img, (0, 0, pad_right, pad_bottom), fill=0)

        return img

    def __getitem__(self, idx):

        if self.mapper is None:
            raw_image = self.h5_file.get("images")[idx]
            label = self.h5_file.get("targets")[idx]
            filename = None
        else:
            dp_mapper = self.mapper.loc[idx]
            raw_image = self.h5_file.get("images")[dp_mapper["h5_index"]]
            label = dp_mapper["target"]
            filename = dp_mapper["filename"]

        img = Image.open(io.BytesIO(raw_image)).convert("RGB")

        dp = self.mask_by_filename.get(filename, None)

        if dp is not None:
            gaze_rel = torch.tensor(
                [dp["centroid"]["x_rel"], dp["centroid"]["y_rel"]],
                dtype=torch.float32,
            )
            area_rel = torch.tensor(dp.get("area_mask_rel", 0.0), dtype=torch.float32)
        else:
            gaze_rel = torch.tensor([0.5, 0.5], dtype=torch.float32)
            area_rel = torch.tensor(0.0, dtype=torch.float32)

    
        W_original, H_original = img.size
        max_side = max(W_original, H_original)

        w_ratio = W_original / max_side
        h_ratio = H_original / max_side

        ratio = torch.tensor([w_ratio, h_ratio], dtype=torch.float32)

        img = self.pad_to_square(img)

        if self.transform is not None:
            img = self.transform(img)

        meta = {
            "gaze": gaze_rel,
            "ratio": ratio,
            "area": area_rel,
        }

        return img, label, meta