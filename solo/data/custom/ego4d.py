import io
import random
from pathlib import Path
from typing import Union, Callable

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from foveation.gaze_crop import GazeCenteredCrop


class Ego4d(Dataset):
    def __init__(
            self,
            root: Union[Path, str],
            time_window: int = 0,
            transform: Callable[[np.ndarray], np.ndarray] = None,
            fov_type = None,
    ):
        self.root = Path(root)
        self.time_window = time_window
        self.transform = transform

        self.foveation = build_foveation(fov_type)

        print("Ego4d init time_window:", time_window)
        print("Ego4d init foveation_type:", fov_type)

        self.hdf5_file = h5py.File(self.root / "ego4d_diverse_subset.h5", "r")
        self.annot = pd.read_parquet(self.root / "annot.parquet")

    def open_image(self, idx: int) -> Image.Image:
        bin_img = self.hdf5_file.get("frames")[idx]
        img = Image.open(io.BytesIO(bin_img)).convert("RGB")
        return img

    def __len__(self) -> int:
        return len(self.annot)

    def __getitem__(self, idx: int):
        dp: pd.Series = self.annot.iloc[idx]
        img = self.open_image(idx)
        
        if self.time_window == 0:
            if self.foveation is not None:
                img = self.foveation(img, dp)
            return self.transform(img, img), -1

        new_video_name, new_idx, try_cpt = "", idx, 0

        while dp.video_name != new_video_name:
            new_idx = idx + random.randint(-self.time_window, self.time_window)
            new_idx = max(0, min(new_idx, len(self) - 1))

            if try_cpt > 5:
                new_idx = idx

            new_dp = self.annot.iloc[new_idx]
            new_video_name = new_dp.video_name
            try_cpt += 1

        img_pair = self.open_image(new_idx) 

        if self.foveation is not Not:
            img = self.foveation(img, dp)
            img_pair = self.foveation(img_pair, new_dp)

        return self.transform(img, img_pair), -1


# maybe move to own "factory" file
def build_foveation(fov_type):
    
    if fov_type is None:
        return None

    if fov_type == "none":
        return None

    if fov_type == "gaze_crop":
        return GazeCenteredCrop(
            crop_size=336
        )

    # insert other types here

    raise ValueError(f"Unknown foveation type: {fov_type}") 
