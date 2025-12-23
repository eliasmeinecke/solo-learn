
import io
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from PIL import Image


ANNOT_PATH = "/home/data/elias/Ego4dDivSubset/annot.parquet"
H5_PATH = "/home/data/elias/Ego4dDivSubset/ego4d_diverse_subset.h5"


i = 100_001

df = pd.read_parquet(ANNOT_PATH)
with h5py.File(H5_PATH, "r") as hf:
    frame = hf.get("frames")[i]
    saliency = hf.get("saliency")[i]

    frame = Image.open(io.BytesIO(frame))

row = df.iloc[i]

flat_max = saliency.argmax()
max_x, max_y = flat_max % 64, flat_max // 64

fig, ax = plt.subplots(1, 2)
ax[0].imshow(np.array(frame))
ax[1].imshow(saliency)
ax[0].axis("off")
ax[1].axis("off")
ax[0].scatter(row.gaze_loc_x, row.gaze_loc_y, c="red", s=50)
ax[1].scatter(max_x, max_y, c="red", s=50)
plt.tight_layout()
plt.savefig("plots/ego4d_example.png", dpi=200)
print("Saved ego4d_example.png")
