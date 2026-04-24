import argparse
from pathlib import Path
import csv

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import PILToTensor
from tqdm import tqdm

from foveation.utils import ImageNetGazeDataset, build_model_and_foveation, T_POST, IMAGENET_VAL_PATH


# PATHS
OUT_DIR = Path("/home/elias/solo-learn/foveation/analysis/outputs/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CORE EVAL
def evaluate(dataset, model, foveation, device, gaze_mode):
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )
    correct = 0
    n = 0
    with torch.no_grad():
        for (img, label, gaze, _) in tqdm(loader):
            img = img.to(device)
            label = label.to(device)
            B, C, H, W = img.shape
            if gaze_mode == "object":
                gaze = gaze.to(device)
                gaze_abs = gaze.clone()
                gaze_abs[:, 0] = gaze[:, 0] * W
                gaze_abs[:, 1] = gaze[:, 1] * H
            else:
                gaze_abs = torch.tensor([[W/2, H/2]], device=device)
            img = foveation(img, gaze_abs)    
            img = T_POST(img)
            logits = model(img)
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            n += 1
    return correct / n


# SAVE
def save_result(row, out_path):
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Saved → {out_path}")


# RUN
def run_eval(dataset_path, model_name, gaze_mode):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Evaluating {model_name} ---")
    print(f"Dataset: {dataset_path}")
    print(f"Gaze mode: {gaze_mode}")
    dataset = ImageNetGazeDataset(dataset_path, transform=PILToTensor())
    model, foveation = build_model_and_foveation(model_name, device)
    acc = evaluate(dataset, model, foveation, device, gaze_mode)
    print(f"{model_name} → {acc:.4f}")
    return {
        "model": model_name,
        "gaze_mode": gaze_mode,
        "accuracy": acc
    }


# MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaze_mode", type=str, default="object", choices=["object", "central"])
    args = parser.parse_args()
    gaze_mode = args.gaze_mode
    # output file
    out_name = f"{gaze_mode}_linear_eval_offline.csv"
    out_path = OUT_DIR / out_name
    models = [
        "base",
        "crop",
        "blur-light",
        "blur-nosal",
        "blur-strong",
        "cm-light",
        "cm-nosal",
        "cm-strong",
    ]
    print(f"==============================")
    for model_name in models:
        result = run_eval(IMAGENET_VAL_PATH, model_name, gaze_mode)
        save_result(result, out_path)
    print(f"==============================")


if __name__ == "__main__":
    main()