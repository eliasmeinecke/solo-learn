import io
import random
from pathlib import Path
from typing import Union, Callable

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from foveation.factory import build_foveation


class Ego4d(Dataset):
    def __init__(
            self,
            root: Union[Path, str],
            time_window: int = 0,
            transform: Callable[[np.ndarray], np.ndarray] = None,
            foveation: dict | None = None,
    ):
        self.root = Path(root)
        self.time_window = time_window
        self.transform = transform

        self.foveation = build_foveation(foveation)

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
        dp: pd.Series = self.annot.iloc[idx]
        img = self.open_image(idx)
        
        saliency = self.hdf5_file.get("saliency")[idx]
        saliency = saliency.astype(np.float32)
        
        if self.time_window == 0:
            if self.foveation is not None:
                img = self.foveation(img, dp, saliency)
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

        if self.foveation is not None:
            img = self.foveation(img, dp, saliency)
            img_pair = self.foveation(img_pair, new_dp, saliency)

        return self.transform(img, img_pair), -1