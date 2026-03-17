from pycocotools import mask as mask_utils
import json
import cv2
import numpy as np
from tqdm import tqdm


def compute_centroid(mask):
    mask_uint = mask.astype(np.uint8)
    m = cv2.moments(mask_uint)

    if m["m00"] == 0:
        return None, None

    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return cx, cy


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


def build_gaze_json(train_or_val):

    if train_or_val not in ["train", "val"]:
        raise ValueError("Choose either train or val.")

    input_path = f"/home/data/elias/imagenet_sam_masks/imagenet_{train_or_val}_masks.json"
    output_path = f"/home/data/elias/imagenet_sam_masks/imagenet_{train_or_val}_gaze_only.json"

    with open(input_path, "r") as f:
        data = json.load(f)

    print("Total masks:", len(data))

    out = {}

    for idx in tqdm(data.keys(), desc=f"Processing {train_or_val}"):

        entry = data[idx]

        mask = mask_utils.decode(entry["rle"])
        H, W = mask.shape

        mask_clean = fill_mask_holes_floodfill(mask)

        cx, cy = compute_centroid(mask_clean)

        # fallback: image center
        if cx is None or cy is None:
            cx = W / 2
            cy = H / 2

        cx_rel = float(cx / W)
        cy_rel = float(cy / H)

        area_mask_rel = float(mask_clean.sum() / (H * W))

        # only necessary output
        out[idx] = {
            "filename": entry["filename"],
            "centroid": {
                "x_rel": cx_rel,
                "y_rel": cy_rel
            },
            "area": area_mask_rel
        }

    with open(output_path, "w") as f:
        json.dump(out, f)

    print("Saved:", output_path)


if __name__ == "__main__":
    print("---------------------------------------------")
    # build_gaze_json("train")
    # build_gaze_json("val")
