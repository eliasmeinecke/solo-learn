
import numpy as np
from PIL import Image
import pandas as pd
        
"""
Inspired by: A NEW FOVEAL CARTESIAN GEOMETRY APPROACH USED FOR OBJECT TRACKING. José Martínez, Leopoldo Altamirano 
"""
class FovealCartesianGeometry():
    def __init__(self, p0=64, alpha=0.5, beta=0.0):
        self.p0 = p0
        self.alpha = alpha
        self.beta = beta

    def inverse_map(self, xp, yp, x0, y0, cx, cy, saliency_map):
        dx = xp - cx
        dy = yp - cy

        p = max(abs(dx), abs(dy))

        if p <= self.p0:
            s = 1.0
        else:
            # only approximation of salient region-value
            xs = int(np.clip(x0 + dx, 0, saliency_map.shape[1] - 1))
            ys = int(np.clip(y0 + dy, 0, saliency_map.shape[0] - 1))

            S_val = saliency_map[ys, xs]

            alpha_eff = self.alpha / (1.0 + self.beta * S_val)
            s = (self.p0 / p) ** alpha_eff

        x = int(dx / s + x0)
        y = int(dy / s + y0)

        return x, y

    def __call__(self, img: Image.Image, annot: pd.Series, saliency: np.ndarray) -> Image.Image:
        img_np = np.array(img)
        H, W, C = img_np.shape

        # normalize saliency just to be safe
        S = saliency.astype(np.float32)
        S = (S - S.min()) / (S.max() - S.min() + 1e-6)

        x0, y0 = annot.gaze_loc_x, annot.gaze_loc_y

        out_size = min(H, W)
        cx = cy = out_size // 2
        out = np.zeros((out_size, out_size, C), dtype=img_np.dtype)

        for yp in range(out_size):
            for xp in range(out_size):
                x, y = self.inverse_map(
                    xp, yp, x0, y0, cx, cy, S
                )
                if 0 <= x < W and 0 <= y < H:
                    out[yp, xp] = img_np[y, x]

        valid_mask = np.any(out != 0, axis=-1)
        ys, xs = np.where(valid_mask)

        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        out = out[y_min:y_max+1, x_min:x_max+1]

        out = Image.fromarray(out)
        
        return out


"""
error that I got (also without saliency):
Original Traceback (most recent call last):
  File "/home/elias/.conda/envs/solo/lib/python3.10/site-packages/torch/utils/data/_utils/worker.py", line 351, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
  File "/home/elias/.conda/envs/solo/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/elias/.conda/envs/solo/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/elias/solo-learn/solo/data/pretrain_dataloader.py", line 57, in __getitem__
    data = super().__getitem__(index)
  File "/home/elias/solo-learn/solo/data/custom/ego4d.py", line 73, in __getitem__
    img = self.foveation(img, dp, saliency)
  File "/home/elias/solo-learn/foveation/methods/fcg.py", line 53, in __call__
    y_min, y_max = ys.min(), ys.max()
  File "/home/elias/.conda/envs/solo/lib/python3.10/site-packages/numpy/_core/_methods.py", line 49, in _amin
    return umr_minimum(a, axis, None, out, keepdims, initial, where)
ValueError: zero-size array to reduction operation minimum which has no identity

problem might be here:
p = max(abs(dx), abs(dy))
if p <= p0:
    s = 1.0
else:
    s = (p0 / p) ** alpha
x = dx / s + x0
y = dy / s + y0

all pixels might get thrown out of the picture?
"""