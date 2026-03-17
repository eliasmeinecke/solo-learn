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


def update_mask_json(train_or_val):
    
    if train_or_val not in ["train", "val"]:
        print("Choose either train or val.")
        return None
    
    # Load JSON
    input_path = f"/home/data/elias/imagenet_sam_masks/imagenet_{train_or_val}_masks.json"
    output_path = f"/home/data/elias/imagenet_sam_masks/imagenet_{train_or_val}_masks_with_center.json"


    with open(input_path, "r") as f:
        data = json.load(f)

    print("Total masks:", len(data))


    # Process all masks
    for idx in tqdm(data.keys()):

        mask = mask_utils.decode(data[idx]["rle"])
        H, W = mask.shape

        mask_clean = fill_mask_holes_floodfill(mask)

        cx, cy = compute_centroid(mask_clean)

        
        # Fallback: image center
        if cx is None or cy is None:
            cx = W / 2
            cy = H / 2

        cx_rel = float(cx / W)
        cy_rel = float(cy / H)
        
        # 1) True mask area
        area_mask_rel = float(mask_clean.sum() / (H * W))

        # 2) Bounding box area
        x1, y1, x2, y2 = data[idx]["bbox"]
        bbox_area = (x2 - x1) * (y2 - y1)
        area_bbox_rel = float(bbox_area / (H * W))
        
        # Write to JSON
        data[idx]["centroid"] = {
            "x_rel": cx_rel,
            "y_rel": cy_rel
        }

        data[idx]["area_mask_rel"] = area_mask_rel
        data[idx]["area_bbox_rel"] = area_bbox_rel


    # Save updated JSON
    with open(output_path, "w") as f:
        json.dump(data, f)

    print("Saved:", output_path)
    
    
if __name__ == "__main__":
    print("---------------------------------------------")
    #update_mask_json("val")
    

   