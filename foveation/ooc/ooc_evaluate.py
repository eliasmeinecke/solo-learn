import argparse
from pathlib import Path
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2 as v2

from foveation.factory import setup_exact_foveation
from foveation.ooc.ooc_utils import load_data, load_model, IdentityFoveation


DATASETS = ["original", "inpainted", "object", "ooc"]
RESULTS_PATH = Path("/home/elias/solo-learn/foveation/analysis/outputs/data/ooc_results.csv")


def tensor_to_img(x):
    x = x.detach().cpu().float()
    if x.ndim == 4:
        x = x[0]
    x = x.permute(1, 2, 0).numpy()
    return np.clip(x / 255.0, 0, 1)


def show_debug_batch(img, gaze, foveation, T_post, model, device, model_name, dataset_name, idx, label):

    img = img.to(device)
    gaze = gaze.to(device)

    # --- gaze rel → abs ---
    B, C, H, W = img.shape
    gaze_abs = gaze.clone()
    gaze_abs[0, 0] *= W
    gaze_abs[0, 1] *= H

    # --- pipeline ---
    img_orig = img.clone()
    img_fov = foveation(img.clone(), gaze_abs.clone())
    img_post = T_post(img_fov.clone())

    logits = model(img_post)
    pred = logits.argmax(dim=1).item()

    # --- to numpy ---
    img_orig_np = tensor_to_img(img_orig)
    img_fov_np = tensor_to_img(img_fov)

    gx = int(gaze_abs[0, 0].item())
    gy = int(gaze_abs[0, 1].item())

    # --- plot ---
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))

    axs[0].imshow(img_orig_np)
    axs[0].scatter(gx, gy, c="red", s=40)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(img_fov_np)
    axs[1].set_title(f"Pred={pred}")
    axs[1].axis("off")

    plt.suptitle(f"GT={label}")

    # --- save path ---
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "debug_plots" / model_name / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"sample_{idx:03d}.png"

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[DEBUG] Saved: {out_path}")
    
    
def evaluate(model, loader, foveation, T_post, device, model_name, dataset_name, save_preds=False, debug=False, n_debug=3):

    model.eval()
    model.to(device)

    correct = 0
    total = 0
    
    all_preds = [] if save_preds else None

    with torch.no_grad():
        for i, (img, label, gaze) in enumerate(loader):

            img = img.to(device)
            label = label.to(device)
            gaze = gaze.to(device)
            
            if debug and i < n_debug:
                show_debug_batch(img, gaze, foveation, T_post, model, device, model_name=model_name, dataset_name=dataset_name, idx=i, label=label.item())
                
            # convert relative gaze to absolute gaze
            B, C, H, W = img.shape
            gaze_abs = gaze.clone()
            gaze_abs[0, 0] *= W
            gaze_abs[0, 1] *= H 

            img = foveation(img, gaze_abs)
            img = T_post(img)

            logits = model(img)
            probs = torch.softmax(logits, dim=1)
            
            top5_probs, top5_idx = torch.topk(probs, k=5, dim=1)

            pred = top5_idx[0, 0].item()
            label_val = label.item()
            correct_sample = int(pred == label_val)

            # --- save ---
            if save_preds:
                all_preds.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "idx": i,

                    "label": label_val,
                    "pred": pred,
                    "correct": correct_sample,

                    "conf_top1": top5_probs[0, 0].item(),
                    "top5_idx": top5_idx[0].tolist(),
                    "top5_probs": top5_probs[0].tolist(),
                })

            correct += correct_sample
            total += 1
    
    if save_preds:
        save_predictions(all_preds, model_name, dataset_name)

    return correct / total


def save_predictions(preds, model_name, dataset_name):

    base_dir = Path("/home/elias/solo-learn/foveation/analysis/outputs/data/ooc_per_sample")
    out_dir = base_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{model_name}_{dataset_name}.csv"

    with open(out_path, "w") as f:
        fieldnames = [
            "model", "dataset", "idx",
            "label", "pred", "correct",
            "conf_top1",
            "top5_idx", "top5_probs"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in preds:
            row_copy = row.copy()

            # lists → json string
            row_copy["top5_idx"] = json.dumps(row_copy["top5_idx"])
            row_copy["top5_probs"] = json.dumps(row_copy["top5_probs"])

            writer.writerow(row_copy)

    print(f"[Saved predictions] {out_path}")


def save_results(model_name, results):

    file_exists = RESULTS_PATH.exists()

    # load existing results
    existing = {}

    if file_exists:
        with open(RESULTS_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row["model"]] = row

    # update / insert
    existing[model_name] = {
        "model": model_name,
        **{d: results[d] for d in DATASETS}
    }

    # write full file
    with open(RESULTS_PATH, "w") as f:
        fieldnames = ["model"] + DATASETS
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for row in existing.values():
            writer.writerow(row)
    
    print(f"[Saved results] {RESULTS_PATH}")
            
            
def load_existing_results():
    if not RESULTS_PATH.exists():
        return {}

    results = {}
    with open(RESULTS_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["model"]] = {
                d: float(row[d]) for d in DATASETS
            }
    return results
            

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_preds", action="store_true")
    args = parser.parse_args()

    existing_results = load_existing_results()
    if args.model in existing_results and not args.force:

        print(f"\n=== Cached Results: {args.model} ===")

        results = existing_results[args.model]

        print("\n===== SUMMARY =====")
        for k, v in results.items():
            print(f"{k:10s}: {v:.4f}")

        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"  # CHANGE THIS! just set because no free GPUs at the moment
    
    # model
    model = load_model(args.model).to(device)

    # foveation
    if args.model in ["base", "dummy"]:
        foveation = IdentityFoveation()
    else:
        foveation = setup_exact_foveation(args.model)

    print(f"Foveation Type: {foveation}")
    
    foveation = foveation.to(device)

    # post transform
    T_post = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    print(f"\n=== Evaluating: {args.model} ===")

    results = {}

    for dataset_name in DATASETS:

        print(f"\n→ Dataset: {dataset_name}")

        loader = load_data(dataset_name)

        acc = evaluate(
            model, loader, foveation, T_post, device, model_name=args.model, dataset_name=dataset_name, save_preds=args.save_preds, debug=args.debug
        )

        results[dataset_name] = acc

        print(f"Accuracy: {acc:.4f}")

    print("\n===== SUMMARY =====")
    for k, v in results.items():
        print(f"{k:10s}: {v:.4f}")
        
    save_results(args.model, results)


if __name__ == "__main__":
    main()