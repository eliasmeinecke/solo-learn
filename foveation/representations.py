import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import PILToTensor
import torchvision.transforms.v2 as v2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from foveation.ooc.ooc_utils import load_model, IdentityFoveation
from foveation.factory import setup_exact_foveation

IMAGENET_VAL_PATH = "/home/data/ILSVRC_real/val"
JSON_PATH = "/home/data/elias/imagenet_sam_masks/imagenet_val_gaze_only.json"

DATA_DIR = Path("/home/elias/solo-learn/foveation/analysis/outputs/data/representations")
DATA_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR = Path("/home/elias/solo-learn/foveation/analysis/outputs/figures/representations")
FIG_DIR.mkdir(parents=True, exist_ok=True)


SELECTED_CLASSES = [207, 281, 294, 340, 386, 407, 817, 569, 404, 880, 954, 963, 953, 950, 934, 429, 508, 559, 907, 414]

CLASS_INDEX_PATH = "/home/elias/solo-learn/imagenet_class_index.json"

with open(CLASS_INDEX_PATH, "r") as f:
    CLASS_INDEX = json.load(f)

# mapping: int → imagenet class name
IDX_TO_NAME = {int(k): v[1] for k, v in CLASS_INDEX.items()}

CLASS_NAME_TO_GROUP = {
    "golden_retriever": "animal",
    "tabby": "animal",
    "brown_bear": "animal",
    "zebra": "animal",
    "African_elephant": "animal",
    "ambulance": "vehicle",
    "sports_car": "vehicle",
    "garbage_truck": "vehicle",
    "airliner": "vehicle",
    "unicycle": "vehicle",
    "banana": "food",
    "pizza": "food",
    "pineapple": "food",
    "orange": "food",
    "hotdog": "food",
    "baseball": "object",
    "computer_keyboard": "object",
    "folding_chair": "object",
    "wine_bottle": "object",
    "backpack": "object",
}
    

class GazeImageNet(Dataset):
    def __init__(self, root, gaze_json, transform=None):

        self.dataset = ImageFolder(root=root)
        self.transform = transform

        with open(gaze_json, "r") as f:
            gaze_data = json.load(f)

        self.mask_by_filename = {v["filename"]: v for v in gaze_data.values()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        path, label = self.dataset.samples[idx]
        filename = Path(path).name

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # --- gaze + area ---
        dp = self.mask_by_filename.get(filename, None)
        if dp is not None:
            gaze_rel = torch.tensor([
                    dp["centroid"]["x_rel"],
                    dp["centroid"]["y_rel"]
                ], dtype=torch.float32)
        else:
            gaze_rel = torch.tensor([0.5, 0.5], dtype=torch.float32)

        return img, label, gaze_rel
    

def extract_features(model, loader, device, model_name, foveation, T_post):

    model.eval()
    model.to(device)

    features = []
    labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Model {model_name}", total=len(loader))

        for img, label, gaze in pbar:

            l = label.item()

            if l not in SELECTED_CLASSES:
                continue

            img = img.to(device)
            gaze = gaze.to(device)

            # --- gaze → absolute ---
            B, C, H, W = img.shape
            gaze_abs = gaze.clone()
            gaze_abs[:, 0] = gaze[:, 0] * W
            gaze_abs[:, 1] = gaze[:, 1] * H

            # --- foveation ---
            img = foveation(img, gaze_abs)

            # --- post transform ---
            img = T_post(img)

            # --- feature extraction ---
            feat = model.backbone(img)

            if feat.ndim == 4:
                feat = feat.mean(dim=[2, 3])

            feat = feat.squeeze().cpu().numpy()

            features.append(feat)
            labels.append(l)

    features = np.array(features)
    labels = np.array(labels)

    print(f"\nExtracted features: {features.shape}")

    # --- save ---
    out_path = DATA_DIR / f"{model_name}_features.npz"

    np.savez_compressed(
        out_path,
        features=features,
        labels=labels
    )

    print(f"Saved → {out_path}")
    

def load_features(model_name):
    path = DATA_DIR / f"{model_name}_features.npz"

    data = np.load(path)

    features = data["features"]
    labels = data["labels"]

    print(f"Loaded {features.shape[0]} samples")

    return features, labels


def filter_classes(features, labels, selected_classes):
    mask = np.isin(labels, selected_classes)
    return features[mask], labels[mask]


def compute_pca(features):
    features = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2)
    return pca.fit_transform(features)


def compute_separation(features, labels):

    features = np.array(features)
    labels = np.array(labels)

    intra_dists = []
    inter_dists = []

    for i in range(len(features)):
        for j in range(i + 1, len(features)):

            dist = np.linalg.norm(features[i] - features[j])

            if labels[i] == labels[j]:
                intra_dists.append(dist)
            else:
                inter_dists.append(dist)

    intra_mean = np.mean(intra_dists)
    inter_mean = np.mean(inter_dists)

    score = inter_mean / intra_mean

    return {
        "intra": intra_mean,
        "inter": inter_mean,
        "ratio": score
    }
    

def plot_single_pca(features_2d, labels, model_name, metrics):

    df = pd.DataFrame({
        "x": features_2d[:, 0],
        "y": features_2d[:, 1],
        "label_idx": labels
    })

    df["label"] = df["label_idx"].map(IDX_TO_NAME)
    df["label"] = df["label"].str.replace("_", " ").str.title()

    plt.figure(figsize=(6, 5))
    sns.set_style("whitegrid")

    palette = sns.color_palette("tab10", n_colors=df["label"].nunique())
    palette_dict = dict(zip(sorted(df["label"].unique()), palette))

    # --- scatter ---
    ax = sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="label",
        palette=palette_dict,
        s=20,
        alpha=0.7,
        edgecolor="none",
        legend=True
    )

    # --- centroids ---
    centroids = df.groupby("label")[["x", "y"]].mean().reset_index()

    for _, row in centroids.iterrows():
        label = row["label"]

        ax.scatter(
            row["x"],
            row["y"],
            color=palette_dict[label],   # SAME COLOR
            s=120,
            marker="X",
            edgecolor="black",
            linewidth=0.5,
            zorder=5
        )

    # --- titles & labels ---
    plt.title(
        f"Feature Space (PCA) – {model_name}\n"
        f"Separation: {metrics['ratio']:.2f}  "
        f"(inter={metrics['inter']:.2f}, intra={metrics['intra']:.2f})",
        fontsize=11
    )
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    # cleaner legend
    plt.legend(
        title="Class",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    plt.tight_layout()

    out_path = FIG_DIR / f"{model_name}_single-pca.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved → {out_path}")
    

def get_group_labels(labels):
    groups = []

    for l in labels:
        class_name = IDX_TO_NAME[l]
        group = CLASS_NAME_TO_GROUP[class_name]
        groups.append(group)

    return np.array(groups)


def compute_group_separation(features, groups):

    intra = []
    inter = []

    for i in range(len(features)):
        for j in range(i + 1, len(features)):

            dist = np.linalg.norm(features[i] - features[j])

            if groups[i] == groups[j]:
                intra.append(dist)
            else:
                inter.append(dist)

    intra_mean = np.mean(intra)
    inter_mean = np.mean(inter)

    return {
        "intra": intra_mean,
        "inter": inter_mean,
        "ratio": inter_mean / intra_mean
    }
    
    
def plot_group_pca(features_2d, labels, model_name, metrics):

    df = pd.DataFrame({
        "x": features_2d[:, 0],
        "y": features_2d[:, 1],
        "label_idx": labels
    })

    df["class"] = df["label_idx"].map(IDX_TO_NAME)
    df["group"] = df["class"].map(CLASS_NAME_TO_GROUP)

    plt.figure(figsize=(6, 5))
    sns.set_style("whitegrid")

    # --- fixed palette ---
    palette = {
        "animal": "#4C72B0",
        "vehicle": "#DD8452",
        "food": "#55A868",
        "object": "#C44E52",
    }

    ax = sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="group",
        palette=palette,
        s=20,
        alpha=0.7,
        edgecolor="none"
    )

    # --- centroids per group ---
    centroids = df.groupby("group")[["x", "y"]].mean().reset_index()

    for _, row in centroids.iterrows():
        g = row["group"]

        ax.scatter(
            row["x"],
            row["y"],
            color=palette[g],
            s=180,
            marker="X",
            edgecolor="black",
            linewidth=0.7,
            zorder=5
        )

    # --- title ---
    plt.title(
        f"Feature Space by Semantic Group – {model_name}\n"
        f"Separation: {metrics['ratio']:.2f}  "
        f"(inter={metrics['inter']:.2f}, intra={metrics['intra']:.2f})",
        fontsize=11
    )

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    plt.legend(
        title="Group",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    plt.tight_layout()

    out_path = FIG_DIR / f"{model_name}_group-pca.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved → {out_path}")
    
    
def main(model_name, task):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    if task == "extract":
        full_model = load_model(model_name)
        # --- foveation ---
        if model_name in ["base", "dummy"]:
            foveation = IdentityFoveation()
        else:
            foveation = setup_exact_foveation(model_name)
        T_post = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])
        dataset = GazeImageNet(root=IMAGENET_VAL_PATH, gaze_json=JSON_PATH, transform=PILToTensor())
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        extract_features(full_model, loader, device, model_name, foveation, T_post)
    elif task == "single-pca":
        features, labels = load_features(model_name)
        # selected subset of 20 classes (golden-retriever, ambulance, banana, wine-bottle)
        selected_subset = [207, 407, 954, 907]
        features, labels = filter_classes(features, labels, selected_subset)
        metrics = compute_separation(features, labels)
        features_2d = compute_pca(features)
        plot_single_pca(features_2d, labels, model_name, metrics)
    elif task == "group-pca":
        features, labels = load_features(model_name)
        groups = get_group_labels(labels)
        metrics = compute_group_separation(features, groups)
        features_2d = compute_pca(features)
        plot_group_pca(features_2d, labels, model_name, metrics)
    else:
        raise ValueError("Invalid task type given!")
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="base")  # valid foveation-types
    parser.add_argument("--task", type=str, default="extract")  # extract, single-pca, group-pca
    args = parser.parse_args()
        
    main(args.model, args.task)
    