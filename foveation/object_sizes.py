import argparse
from pathlib import Path
import csv
from tqdm import tqdm
import torch
import torchvision.transforms.v2 as v2
from torchvision.transforms import PILToTensor
from torch.utils.data import DataLoader

from foveation.factory import setup_exact_foveation
from foveation.ooc.ooc_utils import load_model, IdentityFoveation

from foveation.utils import ImageNetSizeDataset


# -----------------------
# CONFIG
# -----------------------

IMAGENET_VAL_PATH = "/home/data/ILSVRC_real/val"
JSON_PATH = "/home/data/elias/imagenet_sam_masks/imagenet_val_gaze_only.json"

OUT_DIR = Path("/home/elias/solo-learn/foveation/analysis/outputs/data/size_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------
# EVALUATE
# -----------------------

def evaluate(model, loader, foveation, T_post, device, model_name):

    model.eval()
    model.to(device)

    results = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Evaluating {model_name}", total=len(loader))

        for i, (img, label, gaze, area) in enumerate(pbar):

            img = img.to(device)
            label = label.to(device)
            gaze = gaze.to(device)
            area = area.to(device)

            # --- convert gaze ---
            B, C, H, W = img.shape
            gaze_abs = gaze.clone()
            gaze_abs[:, 0] = gaze[:, 0] * W
            gaze_abs[:, 1] = gaze[:, 1] * H

            # --- foveation ---
            img = foveation(img, gaze_abs)

            # --- post ---
            img = T_post(img)

            # --- forward ---
            logits = model(img)
            probs = torch.softmax(logits, dim=1)

            top5_probs, top5_idx = torch.topk(probs, k=5, dim=1)

            pred = top5_idx[:, 0]
            correct = (pred == label)

            results.append({
                "model": model_name,
                "idx": i,

                "label": label.item(),
                "pred": pred.item(),
                "correct": int(correct.item()),

                "conf_top1": top5_probs[0, 0].item(),

                "top5_idx": top5_idx[0].tolist(),
                "top5_probs": top5_probs[0].tolist(),

                "mask_area": area.item(),
            })

    return results


# -----------------------
# SAVE
# -----------------------

def save_results(results, model_name):

    out_path = OUT_DIR / f"{model_name}.csv"

    keys = results[0].keys()

    with open(out_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved → {out_path}")


# -----------------------
# MAIN
# -----------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="base")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ImageNetSizeDataset(
        root=IMAGENET_VAL_PATH,
        gaze_json=JSON_PATH, 
        transform=PILToTensor()
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # --- model ---
    model = load_model(args.model)

    # --- foveation ---
    if args.model in ["base", "dummy"]:
        foveation = IdentityFoveation()
    else:
        foveation = setup_exact_foveation(args.model)

    foveation = foveation.to(device)

    # --- post ---
    T_post = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    print(f"\n=== Evaluating ImageNet-Val: {args.model} ===")

    results = evaluate(model, loader, foveation, T_post, device, args.model)

    save_results(results, args.model)


if __name__ == "__main__":
    main()