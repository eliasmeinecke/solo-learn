import io
import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as TF

from solo.data.custom.base import H5ClassificationDataset


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
        
        self.transform = transform # make sure this is right!
        
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

    
    
# do this when normal one works
class ImgNetWithGaze42(H5ClassificationDataset):

    def __init__(self, root, transform, split, subset):
        super().__init__(root, transform, split, subset)

        # adjust this to 42
        json_path = f"/home/data/elias/imagenet_sam_masks/imagenet_{split}_masks_with_center.json"
        
        with open(json_path, "r") as f:
            data = json.load(f)

        self.mask_by_filename = {
            v["filename"]: v
            for v in data.values()
        }

    def __getitem__(self, idx):

        if self.mapper is None:
            raise RuntimeError("Mapper required for gaze integration")

        dp = self.mapper.loc[idx]

        raw_image = self.h5_file.get("images")[dp["h5_index"]]
        label = dp["target"]
        filename = dp["filename"]

        image = Image.open(io.BytesIO(raw_image)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        gaze_rel = torch.tensor([0.5, 0.5], dtype=torch.float32)

        if filename in self.mask_by_filename:
            mask_json = self.mask_by_filename.get(filename, None)
            
            if mask_json is not None:
                gaze_rel = torch.tensor([
                    mask_json["centroid"]["x_rel"],
                    mask_json["centroid"]["y_rel"]
                ], dtype=torch.float32)
                area_rel = mask_json.get("area_mask_rel", 0.0)
            else:
                # fallback → center
                gaze_rel = torch.tensor([0.5, 0.5], dtype=torch.float32)
                area_rel = 0.0

        return image, label, {"gaze": gaze_rel, "area": area_rel}