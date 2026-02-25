import io
import random
from pathlib import Path
from typing import Union, Callable, Optional

import h5py
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from foveation.methods.gaze_crop import GazeCenteredCrop


class Ego4d(Dataset):
    def __init__(
            self,
            root: Union[Path, str],
            time_window: int = 0,
            transform: Callable[[np.ndarray], np.ndarray] = None,
            foveation: Optional[dict] = None,
    ):
        self.root = Path(root)
        self.time_window = time_window
        self.transform = transform
        self.foveation = foveation
        self.hdf5_file = h5py.File(self.root / "ego4d_diverse_subset.h5", "r")
        self.annot = pd.read_parquet(self.root / "annot.parquet")

    def open_image(self, idx: int) -> Image.Image:
        bin_img = self.hdf5_file.get("frames")[idx]
        img = Image.open(io.BytesIO(bin_img)).convert("RGB")
        # convert from BGR to RGB
        img = np.array(img)[:, :, ::-1]
        img = Image.fromarray(img)
        return img

    def __len__(self) -> int:
        return len(self.annot)

    def __getitem__(self, idx: int):
        
        img = self.open_image(idx)
        dp = self.annot.iloc[idx]
        saliency = self.hdf5_file.get("saliency")[idx].astype(np.float32)
        
        new_idx = idx
        dp_pair = dp
        saliency_pair = saliency
        
        if self.time_window > 0:
            new_video_name, try_cpt = "", 0

            while dp.video_name != new_video_name:
                new_idx = idx + random.randint(-self.time_window, self.time_window)
                new_idx = max(0, min(new_idx, len(self) - 1))

                if try_cpt > 5:
                    new_idx = idx

                new_dp = self.annot.iloc[new_idx]
                new_video_name = new_dp.video_name
                try_cpt += 1
            
            dp_pair = self.annot.iloc[new_idx]
            saliency_pair = self.hdf5_file["saliency"][new_idx].astype(np.float32)

        img_pair = self.open_image(new_idx) 
        
        # CPU crop only
        if self.foveation is not None:
                fov_type = self.foveation.get("type", None)
                if fov_type == "crop":
                    params = self.foveation.get("crop", {})
                    crop = GazeCenteredCrop(**params)
                    img = crop(img, dp)
                    img_pair = crop(img_pair, dp_pair)

        img_out = self.transform(img, img_pair)

        # CPU augmentation mode
        if isinstance(img_out, list):
            return img_out, -1

        # GPU augmentation mode (could be reduced by looking at fov_type, only returning essential info and checking as well later)
        img1, img2 = img_out
        
        gaze = torch.tensor(
            [dp.gaze_loc_x, dp.gaze_loc_y],
            dtype=torch.float32
        )

        gaze_pair = torch.tensor(
            [dp_pair.gaze_loc_x, dp_pair.gaze_loc_y],
            dtype=torch.float32
        )

        saliency = torch.from_numpy(saliency).unsqueeze(0).float()
        saliency_pair = torch.from_numpy(saliency_pair).unsqueeze(0).float()

        return (
            (img1, img2),
            (gaze, gaze_pair),
            (saliency, saliency_pair)
        ), -1 