import pandas as pd
from pathlib import Path
from .collect import collect_wandb
from .utils import (
    save_data,
    extract_foveation_type, 
    add_dataset_category, 
    clean_dataset_names
)


def collect_lr_sweep():

    df = collect_wandb(
        filters={
            "$and": [
                {"group": "updated-lr-sweep"},
                {"state": "finished"}
            ]
        }
    )

    knn_top1_cols = [c for c in df.columns if "knn/" in c and "_top1" in c]
    knn_top5_cols = [c for c in df.columns if "knn/" in c and "_top5" in c]

    df["knn_top1_best"] = df[knn_top1_cols].max(axis=1)
    df["knn_top5_best"] = df[knn_top5_cols].max(axis=1)

    if "train_contrastive_loss_epoch" in df.columns:
        df["final_loss"] = df["train_contrastive_loss_epoch"]

    lr_col = "optimizer.lr"  # effective LR
    df["base_lr"] = df[lr_col] / 2  # base LR

    # rename kNN columns
    rename_dict = {}
    for c in knn_top1_cols:
        k = c.split("_")[-2]  # extracts k
        rename_dict[c] = f"knn_top1_k{k}"
    for c in knn_top5_cols:
        k = c.split("_")[-2]
        rename_dict[c] = f"knn_top5_k{k}"
    df = df.rename(columns=rename_dict)

    # select relevant columns
    keep_cols = [
        "name",
        "id",
        lr_col,
        "base_lr",
        "knn_top1_best",
        "knn_top5_best",
        "final_loss"
    ]

    keep_cols += list(rename_dict.values())
    df = df[keep_cols]
    df = df.sort_values("base_lr")
    
    save_data(df, "lr_sweep_processed")


def collect_pretrain():

    df = collect_wandb(
        filters={
            "$and": [
                {"group": "final"},
                {"jobType": "pretrain"},
                {"state": "finished"}
            ]
        }
    )

    df = df[df["name"].str.contains("gpu-aug")]
    df = extract_foveation_type(df)

    knn_top1_cols = [c for c in df.columns if "knn/" in c and "_top1" in c]
    knn_top5_cols = [c for c in df.columns if "knn/" in c and "_top5" in c]

    df["knn_top1_best"] = df[knn_top1_cols].max(axis=1)
    df["knn_top5_best"] = df[knn_top5_cols].max(axis=1)

    if "train_contrastive_loss_epoch" in df.columns:
        df["final_loss"] = df["train_contrastive_loss_epoch"]

    # rename kNN columns
    rename_dict = {}
    for c in knn_top1_cols:
        k = c.split("_")[-2]  # extracts k
        rename_dict[c] = f"knn_top1_k{k}"
    for c in knn_top5_cols:
        k = c.split("_")[-2]
        rename_dict[c] = f"knn_top5_k{k}"
    df = df.rename(columns=rename_dict)

    # select relevant columns
    keep_cols = [
        "name",
        "id",
        "foveation",
        "knn_top1_best",
        "knn_top5_best",
        "final_loss",
    ]

    keep_cols += list(rename_dict.values())
    df = df[keep_cols]
    df = df.sort_values("knn_top1_best", ascending=False)

    save_data(df, "pretrain_processed")


def collect_linear_eval():

    df = collect_wandb(
        filters={
            "$and": [
                {"group": "final"},
                {"jobType": "linear_probe_grid"},
                {"state": "finished"}
            ]
        }
    )

    # remove debug prefix
    df["name_clean"] = df["name"].str.replace("^debug_", "", regex=True)

    # keep only gpu aug runs
    df = df[df["name_clean"].str.contains("gpu-aug")]

    # keep newest run if duplicates exist
    df = df.sort_values("created_at")
    df = df.drop_duplicates(subset="name_clean", keep="last")
    
    # dataset extraction
    df["dataset"] = df["name_clean"].apply(lambda x: x.split("_")[0])
    df = clean_dataset_names(df)
    
    # clean up dataset-names and group in categories
    df = add_dataset_category(df)

    # foveation parsing
    df = extract_foveation_type(df)

    # Linear Eval Accuracy Columns
    acc_cols = [
        c for c in df.columns
        if "max/val/classifier-lr_" in c and "_acc1" in c
    ]

    # best LR result
    df["linear_acc1_best"] = df[acc_cols].max(axis=1)

    df = df[[
        "name_clean",
        "id",
        "dataset",
        "category",
        "foveation",
        "linear_acc1_best"
    ]]

    df = df.sort_values(["category", "dataset", "foveation"])

    save_data(df, "linear_eval_processed")


if __name__ == "__main__":
    print("-------------------------------------------------------------------------------")
    #collect_pretrain()
    #collect_linear_eval()