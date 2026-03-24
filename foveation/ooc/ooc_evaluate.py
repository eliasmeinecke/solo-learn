import argparse
from pathlib import Path
import csv
import torch
import torchvision.transforms.v2 as v2

from foveation.factory import setup_exact_foveation
from foveation.ooc.ooc_utils import load_data, load_model, IdentityFoveation


DATASETS = ["original", "inpainted", "object", "ooc"]
RESULTS_PATH = Path("/home/elias/solo-learn/foveation/analysis/outputs/data/ooc_results.csv")


def evaluate(model, loader, foveation, T_post, device):

    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for img, label, gaze in loader:

            img = img.to(device)
            label = label.to(device)
            gaze = gaze.to(device)

            img = foveation(img, gaze)
            img = T_post(img)

            logits = model(img)
            pred = logits.argmax(dim=1)

            correct += (pred == label).sum().item()
            total += label.size(0)

    return correct / total


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
    args = parser.parse_args()

    existing_results = load_existing_results()
    if args.model in existing_results and not args.force:

        print(f"\n=== Cached Results: {args.model} ===")

        results = existing_results[args.model]

        print("\n===== SUMMARY =====")
        for k, v in results.items():
            print(f"{k:10s}: {v:.4f}")

        return

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"  # CHANGE THIS! just set because no free GPUs at the moment
    
    # model
    model = load_model(args.model).to(device)

    # foveation
    if args.model in ["base", "dummy"]:
        foveation = IdentityFoveation()
    else:
        foveation = setup_exact_foveation(args.model)

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

        acc = evaluate(model, loader, foveation, T_post, device)

        results[dataset_name] = acc

        print(f"Accuracy: {acc:.4f}")

    print("\n===== SUMMARY =====")
    for k, v in results.items():
        print(f"{k:10s}: {v:.4f}")
        
    save_results(args.model, results)


if __name__ == "__main__":
    main()