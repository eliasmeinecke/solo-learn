import random
import csv
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import PILToTensor, InterpolationMode
import torchvision.transforms.v2 as v2

from foveation.utils import build_model_and_foveation
from foveation.crowding.crowding_helper import CrowdingDataset
    
    
# remove central crop from transform! this is important to not cut out flankers.
T_POST_CROWDING = v2.Compose([
        v2.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
        v2.ToImage(),
        v2.ToDtype(
            torch.float32,
            scale=True
        ),
        v2.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def evaluate_crowding(model, device, model_name, foveation, seeds):
    model.eval()
    model.to(device)
    conditions = ["a", "ax", "xax"]
    distances = [5, 10, 20, 30, 50]
    results = []
    for d in distances:
        print(f"\n=== Distance: {d} ===")
        # accumulators over seeds
        stats = {
            c: {"correct": 0, "conf_sum": 0.0, "n": 0}
            for c in conditions
        }
        for seed in seeds:
            set_seed(seed)
            for cond in conditions:
                ds = CrowdingDataset(
                    transform=PILToTensor(),
                    condition=cond,
                    flanker_distance=d
                )
                loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
                with torch.no_grad():
                    for img, label, gaze in loader:
                        img = img.to(device)
                        label = label.to(device)
                        gaze = gaze.to(device)
                        # gaze → absolute
                        B, C, H, W = img.shape
                        gaze_abs = gaze.clone()
                        gaze_abs[0, 0] *= W
                        gaze_abs[0, 1] *= H
                        img = foveation(img, gaze_abs)
                        img = T_POST_CROWDING(img)
                        logits = model(img)
                        probs = torch.softmax(logits, dim=1)
                        top1 = probs.argmax(dim=1)
                        correct = (top1 == label)
                        stats[cond]["correct"] += int(correct.item())
                        stats[cond]["conf_sum"] += probs[0, top1].item()
                        stats[cond]["n"] += 1
        # --- summarize per distance ---
        row = {
            "foveation": model_name,
            "distance": d,
        }
        for cond in conditions:
            acc = stats[cond]["correct"] / stats[cond]["n"]
            conf = stats[cond]["conf_sum"] / stats[cond]["n"]
            row[f"{cond}_acc"] = acc
            row[f"{cond}_conf"] = conf
        results.append(row)
    return results


def save_results(results, out_path):
    write_header = not Path(out_path).exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(results)
    print(f"Saved → {out_path}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="base")
    args = parser.parse_args()
    results_path = Path("/home/elias/solo-learn/foveation/analysis/outputs/data/crowding_results.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, foveation = build_model_and_foveation(args.model, device)
    seeds = list(range(1, 11))  # 10 seeds
    results = evaluate_crowding(
        model=model,
        device=device,
        model_name=args.model,
        foveation=foveation,
        seeds=seeds
    )
    save_results(results, results_path)
    