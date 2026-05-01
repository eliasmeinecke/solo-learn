import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr

from foveation.utils import load_imagenet_class_map
from .utils import (
    load_analysis_data, 
    set_thesis_style, 
    parse_model,
    get_group,
    get_foveation_palette,
    format_label,
    FOVEATION_ORDER,
    EXACT_FOVEATION_ORDER,
    FOVEATION_PALETTE,
    OOC_DATASET_ORDER,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"
    
def plot_lr_sweep_knn_mean():

    set_thesis_style()
    
    df = load_analysis_data("lr_sweep_processed")

    top1_cols = [c for c in df.columns if "knn_top1_k" in c]
    top5_cols = [c for c in df.columns if "knn_top5_k" in c]

    
    # mean over k values
    df["top1_mean"] = df[top1_cols].mean(axis=1)
    df["top5_mean"] = df[top5_cols].mean(axis=1)
    
    df = df.sort_values("base_lr")

    # long format for seaborn
    plot_df = df.melt(
        id_vars="base_lr",
        value_vars=["top1_mean", "top5_mean"],
        var_name="metric",
        value_name="accuracy"
    )

    plot_df["metric"] = plot_df["metric"].map({
        "top1_mean": "kNN Top-1",
        "top5_mean": "kNN Top-5"
    })

    fig, ax = plt.subplots()

    sns.lineplot(
        data=plot_df,
        x="base_lr",
        y="accuracy",
        hue="metric",
        marker="o",
        linewidth=2,
        markersize=8,
        markeredgewidth=0,
        palette="colorblind",
        ax=ax
    )

    # nicer ticks
    ax.set_xticks(df["base_lr"])
    ax.set_xticklabels([f"{lr:.2g}" for lr in df["base_lr"]])

    ax.set_xlabel("Base Learning Rate")
    ax.set_ylabel("Average kNN Accuracy")
    ax.set_title("Learning Rate Sweep")

    ax.legend(title=None, frameon=False)

    out_path = FIG_DIR / "lr_sweep" / "lr_sweep_knn_mean.pdf"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved figure → {out_path}")


def plot_linear_eval_foveated():
    
    set_thesis_style()
    
    df = load_analysis_data("gaze_linear_eval_processed")
    df = df[df["dataset"] == "Foveated Imagenet"]

    # --- baseline ---
    baseline = df[df["foveation"] == "base"]["linear_acc1_best"].iloc[0]

    # --- delta ---
    df["delta"] = df["linear_acc1_best"] - baseline

    # remove baseline
    df = df[df["foveation"] != "base"].copy()

    # --- clean ordering (IMPORTANT FIX) ---
    order = [f for f in EXACT_FOVEATION_ORDER if f != "base"]
    df["foveation"] = pd.Categorical(df["foveation"], categories=order, ordered=True)
    df = df.sort_values("foveation")

    df["group"] = df["foveation"].apply(get_group)

    # --- plot ---
    plt.figure(figsize=(9, 4))

    sns.barplot(
        data=df,
        x="foveation",
        y="delta",
        hue="group",
        dodge=False,
        palette=FOVEATION_PALETTE,
    )

    plt.ylim(top=df["delta"].max() + 0.5)

    plt.ylabel("Δ Top-1 Accuracy vs Baseline (%)")
    plt.xlabel("Foveation Type")
    plt.title(f"Foveated ImageNet Improvement (Baseline = {baseline:.1f}%)")

    # --- annotations ---
    for i, (_, row) in enumerate(df.iterrows()):
        plt.text(i, row["delta"] + 0.15, f"{row['delta']:+.1f}", ha="center", fontsize=9)

    plt.xticks(rotation=0)

    # --- remove legend ---
    plt.legend().remove()

    plt.tight_layout()

    out_path = FIG_DIR / "gaze_linear_eval" / "gaze_linear_eval_delta.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved figure → {out_path}")
    
    
def plot_linear_eval_foveated_offline():
    
    set_thesis_style()
    # --- load ---
    df = load_analysis_data("object_linear_eval_offline")
    df["model"] = df["model"].str.replace("-nosal", "-medium", regex=False)
    df["accuracy"] = df["accuracy"] * 100
    # --- baseline ---
    baseline = df[df["model"] == "base"]["accuracy"].iloc[0]
    # --- delta ---
    df["delta"] = df["accuracy"] - baseline
    # remove baseline
    df = df[df["model"] != "base"].copy()
    # --- ordering ---
    order = [f for f in EXACT_FOVEATION_ORDER if f != "base"]
    df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)
    df = df.sort_values("model")
    palette = get_foveation_palette()
    # --- plot ---
    plt.figure(figsize=(9, 4))
    sns.barplot(
        data=df,
        x="model",
        y="delta",
        hue="model",
        dodge=False,
        palette=palette,
    )
    plt.ylim(top=df["delta"].max() + 0.5)
    plt.ylabel("Δ Top-1 Accuracy vs Baseline (%)")
    plt.xlabel("Foveation Type")
    plt.title(f"Foveated ImageNet Improvement (Baseline = {baseline:.2f}%)")
    # --- annotations ---
    for i, (_, row) in enumerate(df.iterrows()):
        if row["delta"] > 0:
            plt.text(i, row["delta"] + 0.1, f"{row['delta']:+.2f}", ha="center", fontsize=9)
        else:
            plt.text(i, row["delta"] + 0.92, f"{row['delta']:+.2f}", ha="center", fontsize=9)
    plt.xticks(rotation=0)
    plt.axhline(0, linestyle=":", color="gray", linewidth=1, alpha=0.8)
    # --- remove legend ---
    plt.legend().remove()
    plt.tight_layout()
    out_path = FIG_DIR / "gaze_linear_eval" / "object_linear_eval_delta_offline.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved figure → {out_path}")
    
    
def plot_delta_to_central_gaze():
    
    set_thesis_style()

    # --- load ---
    df_gaze = load_analysis_data("gaze_linear_eval_processed")
    df_imnet = load_analysis_data("linear_eval_processed")

    # --- filter ---
    df_gaze = df_gaze[df_gaze["dataset"] == "Foveated Imagenet"]
    df_imnet = df_imnet[df_imnet["dataset"] == "ImageNet-1k 100%"]

    methods = ["base", "crop", "blur", "cm"]

    df_gaze = df_gaze[df_gaze["foveation"].isin(methods)]
    df_imnet = df_imnet[df_imnet["foveation"].isin(methods)]

    # --- merge ---
    df = pd.merge(
        df_gaze[["foveation", "linear_acc1_best"]],
        df_imnet[["foveation", "linear_acc1_best"]],
        on="foveation",
        suffixes=("_gaze", "_imagenet")
    )

    # --- delta ---
    df["delta"] = df["linear_acc1_best_gaze"] - df["linear_acc1_best_imagenet"]

    # remove baseline
    df = df[df["foveation"] != "base"].copy()

    # fix categorical order
    order = [f for f in FOVEATION_ORDER if f != "base"]
    df["foveation"] = pd.Categorical(df["foveation"], categories=order, ordered=True)
    df = df.sort_values("foveation")

    # --- plot ---
    plt.figure(figsize=(7, 4))

    sns.barplot(
        data=df,
        x="foveation",
        y="delta",
        palette=FOVEATION_PALETTE
    )

    plt.ylabel("Δ Accuracy (Object – Central Gaze) (%)")
    plt.xlabel("Foveation Type")
    plt.title("Object vs Central Gaze Performance")

    # --- annotations ---
    for i, (_, row) in enumerate(df.iterrows()):
        plt.text(i, row["delta"] + 0.15, f"{row['delta']:+.1f}", ha="center", fontsize=9)

    plt.xticks(rotation=0)

    # nicer limits (handle negatives too!)
    ymin = min(0, df["delta"].min() - 0.5)
    ymax = df["delta"].max() + 0.5
    plt.ylim(ymin, ymax)

    plt.tight_layout()

    out_path = FIG_DIR / "gaze_linear_eval" / "central_gaze_delta.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved figure → {out_path}")
    
    
def plot_delta_to_central_gaze_offline():

    set_thesis_style()
    # --- load ---
    df_obj = load_analysis_data("object_linear_eval_offline")
    df_obj["model"] = df_obj["model"].str.replace("-nosal", "-medium", regex=False)
    df_cen = load_analysis_data("central_linear_eval_offline")
    df_cen["model"] = df_cen["model"].str.replace("-nosal", "-medium", regex=False)
    # --- to percent ---
    df_obj["accuracy"] *= 100
    df_cen["accuracy"] *= 100
    # --- merge ---
    df = pd.merge(
        df_obj[["model", "accuracy"]],
        df_cen[["model", "accuracy"]],
        on="model",
        suffixes=("_object", "_central")
    )
    # --- delta ---
    df["delta"] = df["accuracy_object"] - df["accuracy_central"]
    # remove baseline
    df = df[df["model"] != "base"].copy()
    # --- ordering ---
    order = [f for f in EXACT_FOVEATION_ORDER if f != "base"]
    df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)
    df = df.sort_values("model")
    # --- plot ---
    palette = get_foveation_palette()
    plt.figure(figsize=(8, 4))
    sns.barplot(
        data=df,
        x="model",
        y="delta",
        palette=palette
    )
    plt.ylabel("Δ Accuracy (Object – Central Gaze) (%)")
    plt.xlabel("Foveation Type")
    plt.title("Object vs Central Gaze Performance")
    # --- annotations ---
    for i, (_, row) in enumerate(df.iterrows()):
        plt.text(i, row["delta"] + 0.15, f"{row['delta']:+.2f}", ha="center", fontsize=9)
    plt.xticks(rotation=0)
    # --- limits ---
    ymin = min(0, df["delta"].min() - 0.5)
    ymax = df["delta"].max() + 0.5
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    out_path = FIG_DIR / "gaze_linear_eval" / "central_gaze_delta_offline.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved figure → {out_path}")
    
    
def plot_ooc_delta_heatmap():

    set_thesis_style()

    df = load_analysis_data("full_ooc_results")

    df["foveation"] = df["foveation"].str.replace("-nosal", "-medium", regex=False)
    df_plot = df[["foveation", "original_acc", "object_acc", "ooc_acc"]].copy()
    df_plot = df_plot.rename(columns={"original_acc": "Original", "object_acc": "Object-Only", "ooc_acc": "OOC"})

    # --- baseline ---
    baseline = df_plot[df_plot["foveation"] == "base"].iloc[0]

    # --- compute delta ---
    df_delta = df_plot.copy()
    for col in OOC_DATASET_ORDER:
        df_delta[col] = df_plot[col] - baseline[col]

    df_delta = df_delta.set_index("foveation")[OOC_DATASET_ORDER]

    # remove baseline row
    df_delta = df_delta.drop(index="base")

    # enforce order (without base)
    order = [m for m in EXACT_FOVEATION_ORDER if m != "base"]
    df_delta = df_delta.loc[order]

    # --- scale to % ---
    df_plot = df_delta * 100

    plt.figure(figsize=(7, 4))

    ax = sns.heatmap(
        df_plot,
        annot=True,
        fmt="+.1f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,   # subtle separation
        linecolor="white",
        cbar_kws={"label": "Δ Accuracy (%)"}
    )

    # remove background grid from style
    ax.grid(False)
    
    # --- title ---
    plt.title("OOC Improvements over Baseline", fontsize=12, pad=60)

    # --- baseline row (aligned, with background) ---
    for j, col in enumerate(OOC_DATASET_ORDER):
        val = baseline[col] * 100
        ax.text(
            j + 0.5,
            -0.8,   # moved slightly down
            f"{val:.1f}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                facecolor="#F0F0F0",
                edgecolor="grey",
                boxstyle="round,pad=0.3"
            )
            # try bbox=dict(facecolor="lightgray", alpha=0.3, edgecolor="none")
        )

    ax.text(
        len(OOC_DATASET_ORDER) / 2,   
        -1.5,                        
        "Baseline Accuracy (%)",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#444444"
    )

    plt.xlabel("Dataset")
    plt.ylabel("Model")

    plt.tight_layout()

    out_path = FIG_DIR / "ooc_eval" / "ooc_delta_heatmap.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved → {out_path}")
    
    
def plot_inpainted_trend():
    
    set_thesis_style()

    df = load_analysis_data("ooc_results")
    
    df = df[~df["model"].str.contains("__")]

    df["model"] = df["model"].str.replace("-nosal", "", regex=False)

    df = df[["model", "inpainted"]].copy()

    df[["type", "strength"]] = df["model"].apply(
        lambda x: pd.Series(parse_model(x))
    )

    # remove base/crop from lines
    df_lines = df[df["type"].isin(["blur", "cm"])]

    # scale to %
    df_plot = df_lines.copy()
    df_plot["inpainted"] = df_plot["inpainted"] * 100

    # --- plot ---
    plt.figure(figsize=(6, 4))
    
    sns.lineplot(
        data=df_plot,
        x="strength",
        y="inpainted",
        hue="type",
        palette=FOVEATION_PALETTE,
        marker="o",
        errorbar=None
    )

    # --- constant lines ---
    base_val = df[df["model"] == "base"]["inpainted"].values[0] * 100
    plt.axhline(base_val, linestyle="--", color=FOVEATION_PALETTE["base"], label="base")
    crop_val = df[df["model"] == "crop"]["inpainted"].values[0] * 100
    plt.axhline(crop_val, linestyle="--", color=FOVEATION_PALETTE["crop"], label="crop")

    # --- y-limits ---
    y_min = min(df_plot["inpainted"].min(), base_val, crop_val)
    y_max = max(df_plot["inpainted"].max(), base_val, crop_val)

    plt.ylim(y_min * 0.95, y_max * 1.05)
    
    # --- x-axis ---
    plt.xticks([2, 3, 4], ["light", "medium", "strong"])

    plt.xlabel("Foveation Strength")
    plt.ylabel("Accuracy (%)")
    plt.title("Performance on Background-Only Images")

    plt.legend(frameon=True)
    plt.tight_layout()

    out_path = FIG_DIR / "ooc_eval" / "inpainted_trend.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved → {out_path}")
    

def build_master_csv():
    inpainted_path = Path("/home/elias/solo-learn/foveation/analysis/outputs/data/ooc_per_sample/base/base_inpainted.csv")
    meta_path = Path("/home/data/elias/ImageNet-OOC1k_flattened/metadata.csv")
    out_path  = Path("/home/elias/solo-learn/foveation/analysis/outputs/data/inpainted_analysis/master.csv")
    # --- load ---
    df_base = pd.read_csv(inpainted_path)
    df_meta = pd.read_csv(meta_path)
    # sanity check
    assert len(df_meta) > df_base["idx"].max(), "Metadata shorter than idx range!"
    # --- align metadata via idx ---
    # take only rows that correspond to predictions
    df_meta_subset = df_meta.iloc[df_base["idx"]].reset_index(drop=True)
    # --- merge (column-wise, safe because of alignment) ---
    df_master = pd.concat([df_base.reset_index(drop=True), df_meta_subset], axis=1)
    # optional: drop redundant columns
    # (class_index ist oft gleich label → kannst du behalten oder entfernen)
    # df_master = df_master.drop(columns=["class_index"])
    # --- save ---
    df_master.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    # quick sanity print
    print(df_master.head())
    
    
def compute_background_stats(min_samples=15):

    master_path  = Path("/home/elias/solo-learn/foveation/analysis/outputs/data/inpainted_analysis/master.csv")
    out_path_cat  = Path("/home/elias/solo-learn/foveation/analysis/outputs/data/inpainted_analysis/background_category_stats.csv")
    out_path_sub  = Path("/home/elias/solo-learn/foveation/analysis/outputs/data/inpainted_analysis/background_subcategory_stats.csv")
    
    df = pd.read_csv(master_path)
    # 1. BACKGROUND CATEGORY
    df_cat = (
        df.groupby("background_category")
        .agg(
            accuracy=("correct", "mean"),
            n_samples=("correct", "count"),
            mean_conf=("conf_top1", "mean"),
        )
        .reset_index()
    )
    # sort (optional, nicer)
    order = ["nature", "human_related", "misc"]
    df_cat["background_category"] = pd.Categorical(
        df_cat["background_category"], categories=order, ordered=True
    )
    df_cat = df_cat.sort_values("background_category")
    df_cat.to_csv(out_path_cat, index=False)
    print(f"Saved → {out_path_cat}")
    # 2. BACKGROUND SUBCATEGORY
    # remove missing / misc (no subcategories there)
    df_sub = df.dropna(subset=["background_subcategory"]).copy()
    df_sub = (
        df_sub.groupby(["background_category", "background_subcategory"])
        .agg(
            accuracy=("correct", "mean"),
            n_samples=("correct", "count"),
            mean_conf=("conf_top1", "mean"),
        )
        .reset_index()
    )
    # filter small groups (important!)
    df_sub = df_sub[df_sub["n_samples"] >= min_samples]
    # sort by accuracy (nice for inspection)
    df_sub = df_sub.sort_values("accuracy", ascending=False)
    df_sub.to_csv(out_path_sub, index=False)
    print(f"Saved → {out_path_sub}")
    
    
def compute_top_predictions(top_k=5, min_samples=15):
    
    master_path  = Path("/home/elias/solo-learn/foveation/analysis/outputs/data/inpainted_analysis/master.csv")
    out_path_cat  = Path("/home/elias/solo-learn/foveation/analysis/outputs/data/inpainted_analysis/top5_predictions_per_category.csv")
    out_path_sub  = Path("/home/elias/solo-learn/foveation/analysis/outputs/data/inpainted_analysis/top5_predictions_per_subcategory.csv")
    idx_class_map = load_imagenet_class_map() 
    df = pd.read_csv(master_path)
    df["pred_name"] = df["pred"].map(idx_class_map)
    # -----------------------------------
    # 1. TOP PREDICTIONS PER CATEGORY
    # -----------------------------------
    cat_results = []
    for cat, group in df.groupby("background_category"):
        counts = group["pred_name"].value_counts()
        top = counts.head(top_k)
        for rank, (cls, cnt) in enumerate(top.items(), start=1):
            cat_results.append({
                "background_category": cat,
                "pred_class": cls,
                "count": cnt,
                "rank": rank
            })
    df_cat = pd.DataFrame(cat_results)
    df_cat.to_csv(out_path_cat, index=False)
    print(f"Saved → {out_path_cat}")
    # -----------------------------------
    # 2. TOP PREDICTIONS PER SUBCATEGORY
    # -----------------------------------
    sub_results = []
    # remove missing subcategories
    df_sub = df.dropna(subset=["background_subcategory"]).copy()
    for (cat, sub), group in df_sub.groupby(["background_category", "background_subcategory"]):
        if len(group) < min_samples:
            continue
        counts = group["pred_name"].value_counts()
        top = counts.head(top_k)
        for rank, (cls, cnt) in enumerate(top.items(), start=1):
            sub_results.append({
                "background_category": cat,
                "background_subcategory": sub,
                "pred_class": cls,
                "count": cnt,
                "rank": rank
            })
    df_sub_out = pd.DataFrame(sub_results)
    df_sub_out.to_csv(out_path_sub, index=False)
    print(f"Saved → {out_path_sub}")
    

def plot_color_std_object_only():

    set_thesis_style()

    df = load_analysis_data("ooc_results")

    # --- split model ---
    df["base_model"] = df["model"].str.split("__").str[0]
    df["variant"] = df["model"].str.split("__").str[1]
    df["variant"] = df["variant"].fillna("default")

    # --- only background variants ---
    valid_variants = ["default", "black", "gray", "white"]
    df = df[df["variant"].isin(valid_variants)]

    # --- clean names ---
    df["base_model"] = df["base_model"].str.replace("-nosal", "-medium", regex=False)

    # --- compute std ---
    df_std = (
        df.groupby("base_model")["object"]
        .std()
        .reset_index()
    )

    # scale to %
    df_std["std"] = df_std["object"] * 100

    # --- order ---
    df_std["base_model"] = pd.Categorical(
        df_std["base_model"],
        categories=EXACT_FOVEATION_ORDER,
        ordered=True
    )
    df_std = df_std.sort_values("base_model")

    palette = get_foveation_palette()

    # --- plot ---
    plt.figure(figsize=(8, 4))

    sns.barplot(
        data=df_std,
        x="base_model",
        y="std",
        hue="base_model",
        palette=palette,
        dodge=False
    )

    # value labels
    for i, v in enumerate(df_std["std"]):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    plt.legend().remove()
    
    plt.ylabel("Std of Accuracy (%)")
    plt.xlabel("Foveation Type")
    plt.title("Background-Color Sensitivity (Object-Only Images)")

    plt.tight_layout()

    out_path = FIG_DIR / "ooc_eval" / "color_std_obj.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved → {out_path}")
    
    
def plot_object_or_ooc_trend(variant):
    
    set_thesis_style()
    
    if variant not in ["Object-Only", "OOC"]:
        raise ValueError("Invalid dataset variant given.")

    df = load_analysis_data("full_ooc_results")
    
    df = df[["foveation", "original_acc", "object_acc", "ooc_acc"]].copy()
    df = df.rename(columns={"original_acc": "Original", "object_acc": "Object-Only", "ooc_acc": "OOC"})
    df["foveation"] = df["foveation"].str.replace("-nosal", "", regex=False)
    

    df = df[["foveation", variant]].copy()

    df[["type", "strength"]] = df["foveation"].apply(
        lambda x: pd.Series(parse_model(x))
    )

    # --- only blur + cm lines ---
    df_lines = df[df["type"].isin(["blur", "cm"])]

    # --- scale ---
    df_plot = df_lines.copy()
    df_plot[variant] *= 100

    # --- plot ---
    plt.figure(figsize=(6, 4))

    sns.lineplot(
        data=df_plot,
        x="strength",
        y=variant,
        hue="type",
        palette=FOVEATION_PALETTE,
        marker="o",
        errorbar=None
    )

    # --- constant lines ---
    base_val = df[df["foveation"] == "base"][variant].values[0] * 100
    crop_val = df[df["foveation"] == "crop"][variant].values[0] * 100

    plt.axhline(base_val, linestyle="--", color=FOVEATION_PALETTE["base"], label="base")
    plt.axhline(crop_val, linestyle="--", color=FOVEATION_PALETTE["crop"], label="crop")

    # --- y-limits ---
    y_min = min(df_plot[variant].min(), base_val, crop_val)
    y_max = max(df_plot[variant].max(), base_val, crop_val)

    plt.ylim(y_min * 0.95, y_max * 1.05)

    # --- x-axis ---
    plt.xticks([2, 3, 4], ["light", "medium", "strong"])

    # --- labels ---
    plt.xlabel("Foveation Strength")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Performance on {variant} Images")

    # --- legend ---
    plt.legend(frameon=True)

    plt.tight_layout()

    out_path = FIG_DIR / "ooc_eval" / f"{variant}_trend.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved → {out_path}")
    
    
def plot_gap(dataset_a, dataset_b):
    
    set_thesis_style()
    
    dataset_map = {"ori": "Original", "obj": "Object-Only", "ooc": "OOC"}
    
    if dataset_a not in dataset_map or dataset_b not in dataset_map:
        raise ValueError("Invalid dataset keys")

    col_a = dataset_map[dataset_a]
    col_b = dataset_map[dataset_b]
    
    df = load_analysis_data("full_ooc_results")
    
    df = df[["foveation", "original_acc", "object_acc", "ooc_acc"]].copy()
    df = df.rename(columns={"original_acc": "Original", "object_acc": "Object-Only", "ooc_acc": "OOC"})
    df["foveation"] = df["foveation"].str.replace("-nosal", "", regex=False)

    # --- compute gap ---
    df["gap"] = df[col_a] - df[col_b]

    df = df[["foveation", "gap"]].copy()

    df[["type", "strength"]] = df["foveation"].apply(
        lambda x: pd.Series(parse_model(x))
    )

    # --- only blur + cm lines ---
    df_lines = df[df["type"].isin(["blur", "cm"])]

    # --- scale ---
    df_plot = df_lines.copy()
    df_plot["gap"] *= 100

    # --- plot ---
    plt.figure(figsize=(6, 4))

    sns.lineplot(
        data=df_plot,
        x="strength",
        y="gap",
        hue="type",
        palette=FOVEATION_PALETTE,
        marker="o",
        errorbar=None
    )

    # --- reference lines ---
    base_gap = df[df["foveation"] == "base"]["gap"].values[0] * 100
    crop_gap = df[df["foveation"] == "crop"]["gap"].values[0] * 100

    plt.axhline(base_gap, linestyle="--", color=FOVEATION_PALETTE["base"], label="base")
    plt.axhline(crop_gap, linestyle="--", color=FOVEATION_PALETTE["crop"], label="crop")

    # --- y-limits ---
    y_min = min(df_plot["gap"].min(), base_gap, crop_gap)
    y_max = max(df_plot["gap"].max(), base_gap, crop_gap)

    plt.ylim(y_min - 0.3, y_max * 1.05)

    # --- x-axis ---
    plt.xticks([2, 3, 4], ["light", "medium", "strong"])

    # --- labels ---
    plt.xlabel("Foveation Strength")
    plt.ylabel(f"{col_a} − {col_b} Accuracy (%)")
    plt.title(f"{col_a} vs {col_b} Gap")

    # --- legend ---
    plt.legend(frameon=True)

    plt.tight_layout()

    # --- filename ---
    out_path = FIG_DIR / "ooc_eval" / f"gap_{dataset_a}_vs_{dataset_b}.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved → {out_path}")
    

def export_ooc_background_summary():

    bg_dir = DATA_DIR / "background_area"
    full_ooc_path = DATA_DIR / "full_ooc_results.csv"

    out_path = bg_dir / "background_summary.csv"

    models = ["crop", "cm-nosal", "cm-strong"]

    df_ooc = pd.read_csv(full_ooc_path)

    rows = []

    for model in models:

        # ------------------
        # background-area data
        # ------------------
        df = pd.read_csv(bg_dir / f"{model}.csv")

        magnif_mean = df["area_ratio"].mean()
        magnif_std  = df["area_ratio"].std()

        fov_mean = df["fov_area"].mean()
        fov_std  = df["fov_area"].std()

        bg_ret = 1 - df["fov_area"]
        bg_mean = bg_ret.mean()
        bg_std  = bg_ret.std()

        orig_mean = df["orig_area"].mean()
        orig_std  = df["orig_area"].std()

        # ------------------
        # OOC gap
        # ------------------
        row_ooc = df_ooc[df_ooc["foveation"] == model].iloc[0]

        object_acc = row_ooc["object_acc"]
        ooc_acc = row_ooc["ooc_acc"]

        ooc_gap = (object_acc - ooc_acc) * 100

        rows.append({
            "model": model,

            "orig_area_mean": orig_mean,
            "orig_area_std": orig_std,

            "fov_area_mean": fov_mean,
            "fov_area_std": fov_std,

            "background_retained_mean": bg_mean,
            "background_retained_std": bg_std,

            "magnification_mean": magnif_mean,
            "magnification_std": magnif_std,

            "object_acc": object_acc,
            "ooc_acc": ooc_acc,
            "ooc_gap_pp": ooc_gap,
        })

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    
    
def plot_model_ooc_confidence():
    
    set_thesis_style()

    models = ["base", "crop", "cm-strong"]

    dfs = []

    for model in models:
        model_path = DATA_DIR / "ooc_per_sample" / model

        for d in ["original", "inpainted", "object", "ooc"]:
            path = model_path / f"{model}_{d}.csv"

            df = pd.read_csv(path)

            df["dataset"] = d
            df["model"] = model

            dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # --- clean ---
    df_all["correct"] = df_all["correct"].astype(bool)
    df_all["model"] = df_all["model"].str.replace("-nosal", "", regex=False)
    
    df_all["model"] = df_all["model"].replace({
        "base": "Base",
        "crop": "Crop",
        "cm-strong": "CM-Strong"
    })
    
    df_all["dataset"] = df_all["dataset"].replace({
        "original": "Original",
        "inpainted": "Background-Only",
        "object": "Object-Only",
        "ooc": "OOC"
    })
    
    models_clean = ["Base", "Crop", "CM-Strong"]
    datasets_clean = ["Original", "Background-Only", "Object-Only", "OOC"]

    palette = {
        True: "#4C72B0",    # blau → korrekt
        False: "#DD8452"    # orange → falsch
    }
    
    # --- plot ---
    g = sns.displot(
        data=df_all,
        x="conf_top1",
        hue="correct",
        palette=palette,
        col="dataset",
        row="model",
        kind="kde",
        bw_adjust=1.5,
        fill=True,
        alpha=0.6,
        common_norm=False,
        height=3,
        aspect=1
    )

    g.set_axis_labels("Confidence", "Density")

    g.fig.subplots_adjust(
        top=0.85,
        wspace=0.15,
        hspace=0.25
    )
    g.set_titles("{row_name} | {col_name}")

    # fix limits
    for ax in g.axes.flat:
        ax.set_xlim(0, 1)
        ax.grid(False)

    # --- add mean confidence annotation per subplot ---
    for i, model in enumerate(models_clean):
        for j, dataset in enumerate(datasets_clean):
            ax = g.axes[i, j]
            subset = df_all[
                (df_all["model"] == model) &
                (df_all["dataset"] == dataset)
            ]
            vals_correct = subset[
                subset["correct"]
            ]["conf_top1"].values
            vals_incorrect = subset[
                ~subset["correct"]
            ]["conf_top1"].values
            mu_correct = vals_correct.mean()
            mu_incorrect = vals_incorrect.mean()
            txt = (
                f"μ(correct)={mu_correct:.2f}\n"
                f"μ(incorrect)={mu_incorrect:.2f}"
            )
            ax.text(
                0.97, 0.97,
                txt,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="none"
                )
            )
    
    g._legend.set_title("Prediction")
    for t, l in zip(g._legend.texts, ["Incorrect", "Correct"]):
        t.set_text(l)

    out_path = FIG_DIR / "ooc_confidence" / "model_confidence_comparison.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved → {out_path}")
    
    
def plot_ooc_confidence_gaps(gap_mode=1):

    set_thesis_style()

    df = load_analysis_data("full_ooc_results")
    df["foveation"] = (df["foveation"].str.replace("-nosal", "-medium", regex=False))
    df = df.rename(columns={"foveation":"model"})
    
    if gap_mode == 1:
        correct_a = "original_conf_correct"
        correct_b = "object_conf_correct"
        incorrect_a = "original_conf_incorrect"
        incorrect_b = "object_conf_incorrect"
        title = "Original → Object-Only"
    elif gap_mode == 2:
        correct_a = "object_conf_correct"
        correct_b = "ooc_conf_correct"
        incorrect_a = "object_conf_incorrect"
        incorrect_b = "ooc_conf_incorrect"
        title = "Object-Only → OOC"
    elif gap_mode == 3:
        correct_a = "original_conf_correct"
        correct_b = "ooc_conf_correct"
        incorrect_a = "original_conf_incorrect"
        incorrect_b = "ooc_conf_incorrect"
        title = "Original → OOC"
    else:
        raise ValueError("gap_mode must be 1,2,3")

    # compute gaps
    df["gap_correct"] = (df[correct_a] - df[correct_b]) * 100
    df["gap_incorrect"] = (df[incorrect_a] - df[incorrect_b]) * 100
    df[["type","strength"]] = df["model"].apply(lambda x: pd.Series(parse_model(x)))
    df_plot = df[df["type"].isin(["blur","cm"])].copy()

    fig, axes = plt.subplots(1,2, figsize=(10,4), sharex=True)
    gap_specs = [("gap_correct", "Correct Predictions"), ("gap_incorrect", "Incorrect Predictions")]

    for ax, (gap_col, panel_title) in zip(axes, gap_specs):

        sns.lineplot(
            data=df_plot,
            x="strength",
            y=gap_col,
            hue="type",
            palette=FOVEATION_PALETTE,
            marker="o", 
            errorbar=None, 
            ax=ax
        )

        base_gap = df[df["model"]=="base"][gap_col].values[0]
        crop_gap = df[df["model"]=="crop"][gap_col].values[0]

        ax.axhline(base_gap, linestyle="--", color=FOVEATION_PALETTE["base"], linewidth=2)
        ax.axhline(crop_gap, linestyle="--", color=FOVEATION_PALETTE["crop"],linewidth=2)
        ax.axhline(0, linestyle=":", color="gray")
        ax.set_xticks([2,3,4])
        ax.set_xticklabels(["light", "medium", "strong"])
        ax.set_xlabel("Foveation Strength")
        ax.set_ylabel("Confidence Drop (%)")
        ax.set_title(panel_title)
        ax.legend().remove()

    legend_elements = [
        Line2D([0], [0], color=FOVEATION_PALETTE["base"], linestyle="--", lw=2, label="base"),
        Line2D([0], [0], color=FOVEATION_PALETTE["crop"], linestyle="--", lw=2, label="crop"),
        Line2D([0], [0], color=FOVEATION_PALETTE["blur"], lw=2, marker="o", label="blur"),
        Line2D([0], [0], color=FOVEATION_PALETTE["cm"], lw=2, marker="o", label="cm"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        frameon=True
    )

    plt.suptitle(f"Confidence Drop from {title}", y=1.02)

    plt.tight_layout()
    out_path = (FIG_DIR / "ooc_confidence" / f"confidence_gap_pair_{gap_mode}.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")
    

def plot_ooc_confidence_dynamics():

    set_thesis_style()
    
    df = load_analysis_data("full_ooc_results")
    
    df["foveation"] = (df["foveation"].str.replace("-nosal", "-medium", regex=False))
    df["group"] = (df["foveation"].apply(get_group))
    
    # average over strengths
    df_plot = (
        df.groupby("group")
        [[
            "original_conf_correct",
            "object_conf_correct",
            "ooc_conf_correct",
            "original_conf_incorrect",
            "object_conf_incorrect",
            "ooc_conf_incorrect"
        ]]
        .mean()
        .reset_index()
    )
    
    group_order = ["base", "crop", "blur", "cm"]
    df_plot["group"] = pd.Categorical(df_plot["group"], categories=group_order, ordered=True)
    df_plot = (df_plot.sort_values("group"))
    
    fig, axes = plt.subplots(1, 2, figsize=(10,4), sharex=True)
    x = [0,1,2]
    conf_specs = [
        (
            [
                "original_conf_correct",
                "object_conf_correct",
                "ooc_conf_correct"
            ],
            "Correct Predictions"
        ),
        (
            [
                "original_conf_incorrect",
                "object_conf_incorrect",
                "ooc_conf_incorrect"
            ],
            "Incorrect Predictions"
        )
    ]
    for ax, (cols, title) in zip(axes, conf_specs):
        for _, row in (df_plot.iterrows()):
            group = row["group"]
            y = [row[cols[0]], row[cols[1]],row[cols[2]]]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                label=group,
                color=FOVEATION_PALETTE[group]
            )
        ax.set_xticks(x)
        ax.set_xticklabels(["Original", "Object-Only", "OOC"])
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Confidence")
        ax.set_title(title)
    
    handles, labels = (axes[0].get_legend_handles_labels())
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=True
    )
    plt.suptitle("Confidence Dynamics Across Context Manipulations",y=1.02)
    plt.tight_layout()
    out_path = (FIG_DIR / "ooc_confidence" / "confidence_dynamics.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")
    
    
def plot_small_objects_accuracy():
    
    set_thesis_style()

    SMALL_THRESH = 0.13229863271117204  # 41% quantile of relative_mask_area following COCO definitions 

    models = [
        "base", "crop",
        "blur-nosal", "blur-light", "blur-strong",
        "cm-nosal", "cm-light", "cm-strong"
    ]
    
    dfs = []

    for model in models:
        path = DATA_DIR / "size_analysis" / f"{model}.csv"
        df = pd.read_csv(path)

        df["model"] = model
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # --- filter small objects ---
    df_small = df_all[df_all["mask_area"] <= SMALL_THRESH].copy()

    # --- compute accuracy per model ---
    df_acc = (
        df_small
        .groupby("model")["correct"]
        .mean()
        .reset_index()
    )
    
    # scale to %
    df_acc["accuracy"] = df_acc["correct"] * 100

    # --- baseline delta ---
    base_acc = df_acc.loc[df_acc["model"]=="base", "accuracy"].iloc[0]
    df_acc["delta"] = df_acc["accuracy"] - base_acc

    # clean names
    df_acc["model"] = df_acc["model"].str.replace("-nosal", "-medium", regex=False)
    df_acc = df_acc[df_acc["model"] != "base"]

    # enforce order
    df_acc["model"] = pd.Categorical(
        df_acc["model"],
        categories=[m for m in EXACT_FOVEATION_ORDER if m != "base"],
        ordered=True
    )
    df_acc = df_acc.sort_values("model")

    palette = get_foveation_palette()
    
    # --- plot ---
    plt.figure(figsize=(7, 4))

    sns.barplot(
        data=df_acc,
        x="model",
        y="delta",
        hue="model",
        palette=palette
    )

    # value labels
    for i, v in enumerate(df_acc["delta"]):
        if v >= 0:
            y = v + 0.35
            va = "bottom"
        else:
            y = v + 1.4
            va = "top"
        plt.text(
            i,
            y,
            f"{v:+.1f}",
            ha="center",
            va=va,
            fontsize=9
        )

    plt.ylabel(f"Δ Accuracy vs Baseline ({base_acc:.3f}%)")
    plt.xlabel("Model")
    plt.title(f"Performance on Small Objects (Rel. Mask Area ≤ {SMALL_THRESH:.3f})")
    
    plt.axhline(0, linestyle=":", linewidth=1, alpha=0.8, color="gray")
    plt.xticks(rotation=0)
    plt.ylim(min(-1, df_acc["delta"].min()-0.5), df_acc["delta"].max() + 1.5)
    
    plt.legend().remove()

    plt.tight_layout()

    out_path = FIG_DIR / "size_analysis" / "small_objects_accuracy.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved → {out_path}")
    

def compute_size_correlations():
    base_path = Path("/home/elias/solo-learn/foveation/analysis/outputs/data/size_analysis")
    model_csvs = ["base.csv", "crop.csv", "blur-light.csv", "blur-nosal.csv", "blur-strong.csv", "cm-light.csv", "cm-nosal.csv", "cm-strong.csv"]
    results = []
    for m in model_csvs:
        df = pd.read_csv(base_path / m)
        # --- core variables ---
        mask_area = df["mask_area"]
        correct = df["correct"]
        # --- spearman ---
        rho, p = spearmanr(mask_area, correct)
        results.append({
            "model": df["model"].iloc[0],
            "spearman_rho": rho,
            "p_value": p,
            "n_samples": len(df),
            "rho_squared": rho ** 2
        })
        print(f"{df['model'].iloc[0]} → rho={rho:.4f}")

    # --- save ---
    out_path = base_path / "size_correlations.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    
    
def plot_accuracy_vs_size():
    
    set_thesis_style()

    models = ["base", "crop", "blur-nosal", "cm-nosal"]

    dfs = []

    for model in models:
        path = DATA_DIR / "size_analysis" / f"{model}.csv"
        df = pd.read_csv(path)

        df["model"] = model
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # --- clean names ---
    df_all["model"] = df_all["model"].str.replace("-nosal", "", regex=False)

    # --- binning ---
    n_bins = 10
    df_all["size_bin"] = pd.qcut(df_all["mask_area"], q=n_bins, duplicates="drop")

    # compute bin centers
    df_all["bin_center"] = df_all["size_bin"].apply(lambda x: x.mid)

    # --- aggregate ---
    df_plot = (
        df_all
        .groupby(["model", "bin_center"])["correct"]
        .mean()
        .reset_index()
    )

    df_plot["accuracy"] = df_plot["correct"] * 100

    # --- plot ---
    plt.figure(figsize=(7, 4))

    sns.lineplot(
        data=df_plot,
        x="bin_center",
        y="accuracy",
        hue="model",
        style="model",
        palette=FOVEATION_PALETTE,
        marker="o",
        errorbar=None
    )
    
    plt.xlabel("Object Size (Rel. Mask Area)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Object Size")

    plt.legend(frameon=True)
    
    plt.tight_layout()

    out_path = FIG_DIR / "size_analysis" / "accuracy_vs_size.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved → {out_path}")
    
    
def plot_accuracy_vs_size_by_strength(method="blur"):
    
    set_thesis_style()

    models = [
        "base",
        "blur-nosal", "blur-light", "blur-strong",
        "cm-nosal", "cm-light", "cm-strong"
    ]

    dfs = []

    for model in models:
        path = DATA_DIR / "size_analysis" / f"{model}.csv"
        df = pd.read_csv(path)

        df["model"] = model
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # --- clean ---
    df_all["model"] = df_all["model"].str.replace("-nosal", "", regex=False)

    # --- parse ---
    df_all[["type", "strength"]] = df_all["model"].apply(
        lambda x: pd.Series(parse_model(x))
    )

    # --- filter method ---
    df_all = df_all[(df_all["type"] == method) | (df_all["model"]=="base")]

    # --- map strength labels ---
    strength_map = {
        2: "light",
        3: "medium",
        4: "strong"
    }
    df_all["strength_label"] = df_all["strength"].map(strength_map).fillna("base")

    # --- binning ---
    n_bins = 10
    df_all["size_bin"] = pd.qcut(df_all["mask_area"], q=n_bins, duplicates="drop")
    df_all["bin_center"] = df_all["size_bin"].apply(lambda x: x.mid)

    # --- aggregate ---
    df_plot = (
        df_all
        .groupby(["model","strength_label","bin_center"])["correct"]
        .mean()
        .reset_index()
    )

    df_plot["accuracy"] = df_plot["correct"] * 100
    # --- compute baseline per bin ---
    base_df = (
        df_plot[df_plot["model"]=="base"]
        [["bin_center","accuracy"]]
        .rename(columns={"accuracy":"base_acc"})
    )

    df_plot = df_plot.merge(base_df, on="bin_center")

    df_plot["delta"] = (df_plot["accuracy"] - df_plot["base_acc"])

    # remove baseline itself from plot
    df_plot = df_plot[df_plot["model"]!="base"]

    # --- ordering ---
    strength_order = ["light", "medium", "strong"]
    df_plot["strength_label"] = pd.Categorical(
        df_plot["strength_label"],
        categories=strength_order,
        ordered=True
    )

    # --- colors (consistent gradient) ---
    palette = {
        "light": "#A3C9A8",
        "medium": "#4C956C",
        "strong": "#1B4332"
    } if method == "blur" else {
        "light": "#CDB4DB",
        "medium": "#9D4EDD",
        "strong": "#5A189A"
    }

    # --- plot ---
    plt.figure(figsize=(7, 4))

    sns.lineplot(
        data=df_plot,
        x="bin_center",
        y="delta",
        hue="strength_label",
        palette=palette,
        marker="o"
    )

    plt.xlabel("Object Size (Rel. Mask Area)")
    plt.ylabel("Δ Accuracy vs Baseline (%)")
    plt.title(f"{method.upper()}")
    
    plt.axhline(0, linestyle=":", linewidth=1, alpha=0.8, color="gray")
    plt.legend(frameon=True)
    plt.tight_layout()
    
    out_path = FIG_DIR / "size_analysis" / f"{method}_size_vs_accuracy.pdf"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved → {out_path}")
    

def plot_crowding_distance_trend(condition="ax", normalize=True):
    
    set_thesis_style()

    df = load_analysis_data("crowding_results")
    df["foveation"] = df["foveation"].str.replace("-nosal", "-medium", regex=False)
    # --- select relevant columns ---
    acc_col = f"{condition}_acc"
    # --- optional normalization ---
    if normalize:
        df["acc_norm"] = df.groupby("foveation")[acc_col].transform(
            lambda x: x / x.iloc[0]  # normalize by distance=5
        )
        y = "acc_norm"
        ylabel = f"Normalized Accuracy ({condition})"
    else:
        y = acc_col
        ylabel = f"Accuracy ({condition})"
    palette = get_foveation_palette()
    # --- plot ---
    plt.figure(figsize=(7, 4))
    sns.lineplot(
        data=df,
        x="distance",
        y=y,
        hue="foveation",
        palette=palette,
        marker="o"
    )
    plt.xlabel("Flanker Distance")
    plt.ylabel(ylabel)
    plt.title(f"Accuracy Trend vs Distance ({condition})")
    if condition == "xax":
        plt.legend().remove()
    else:
        plt.legend(frameon=True)
    plt.tight_layout()
    
    out_path = FIG_DIR / "crowding" / f"{condition}_fdistance_trend.pdf"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved → {out_path}")
    
    
def plot_crowding_intro(distance=20):
    set_thesis_style()
    df = load_analysis_data("crowding_results")
    df["foveation"] = df["foveation"].str.replace("-nosal", "-medium", regex=False)
    df = df[df["distance"] == distance].copy()
    # --- prepare absolute ---
    df_abs = df.copy()
    df_abs["a_acc"] *= 100
    df_abs["foveation"] = pd.Categorical(
        df_abs["foveation"],
        categories=EXACT_FOVEATION_ORDER,
        ordered=True
    )
    df_abs = df_abs.sort_values("foveation")
    df_abs["foveation_label"] = df_abs["foveation"].apply(format_label)
    # --- prepare normalized ---
    keep = ["base", "crop", "blur-light", "cm-light"]
    df_rel = df[df["foveation"].isin(keep)].copy()
    df_rel["a_norm"] = 1.0
    df_rel["ax_norm"] = df_rel["ax_acc"] / df_rel["a_acc"]
    df_rel["xax_norm"] = df_rel["xax_acc"] / df_rel["a_acc"]
    df_long = df_rel.melt(
        id_vars=["foveation"],
        value_vars=["a_norm", "ax_norm", "xax_norm"],
        var_name="condition",
        value_name="accuracy"
    )
    df_long["condition"] = df_long["condition"].str.replace("_norm", "")
    df_long["condition"] = pd.Categorical(
        df_long["condition"],
        categories=["a", "ax", "xax"],
        ordered=True
    )
    palette = get_foveation_palette()
    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 4),
        gridspec_kw={"width_ratios": [1.6, 1]} 
    )
    # LEFT: ABSOLUTE BARPLOT
    ax = axes[0]
    sns.barplot(
        data=df_abs,
        x="foveation_label",
        y="a_acc",
        hue="foveation",
        palette=palette,
        ax=ax
    )
    ax.set_title("Absolute Performance (Condition = a)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("")
    ax.legend().remove()
    for i, (_, row) in enumerate(df_abs.iterrows()):
        ax.text(i, row["a_acc"] + 0.8, f"{row['a_acc']:.1f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, df_abs["a_acc"].max() + 5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    # RIGHT: RELATIVE LINEPLOT
    ax = axes[1]
    sns.lineplot(
        data=df_long,
        x="condition",
        y="accuracy",
        hue="foveation",
        palette=palette,
        marker="o",
        ax=ax
    )
    ax.set_title(f"Crowding Effect (Distance = {distance})")
    ax.set_ylabel("Relative Accuracy")
    ax.set_xlabel("Condition")
    ax.set_ylim(0.6, 1.05)
    ax.legend(frameon=True)
    plt.tight_layout()
    out_path = FIG_DIR / "crowding" / f"crowding_intro_d{distance}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")
    

def plot_crowding_vs_strength(distance=20, mode="a_xax"):
    """
    modes: "a_ax", "ax_xax", "a_xax"
    """
    set_thesis_style()
    df = load_analysis_data("crowding_results")
    df["foveation"] = df["foveation"].str.replace("-nosal", "-medium", regex=False)
    df = df[df["distance"] == distance].copy()
    # --- normalize ---
    df["ax_rel"] = df["ax_acc"] / df["a_acc"]
    df["xax_rel"] = df["xax_acc"] / df["a_acc"]
    # --- compute crowding effect ---
    if mode == "a_ax":
        df["crowding"] = df["ax_rel"] - 1
        title = "Crowding Effect (a → ax)"
    elif mode == "ax_xax":
        df["crowding"] = df["xax_rel"] - df["ax_rel"]
        title = "Crowding Effect (ax → xax)"
    elif mode == "a_xax":
        df["crowding"] = df["xax_rel"] - 1
        title = "Crowding Effect (a → xax)"
    else:
        raise ValueError(mode)
    # --- extract strength ---
    def parse_strength(name):
        if "light" in name:
            return "light"
        elif "strong" in name:
            return "strong"
        elif "blur" in name or "cm" in name:
            return "medium"
        else:
            return "constant"
    df["strength"] = df["foveation"].apply(parse_strength)
    # ordering
    order = ["constant", "light", "medium", "strong"]
    df["strength"] = pd.Categorical(df["strength"], categories=order, ordered=True)
    # --- plot ---
    plt.figure(figsize=(6, 4))
    sns.stripplot(
        data=df,
        x="strength",
        y="crowding",
        hue="foveation",
        palette=get_foveation_palette(),
        size=8,
        jitter=False
    )
    plt.axhline(0, linestyle=":", color="gray", linewidth=1)
    plt.xlabel("Foveation Strength")
    plt.ylabel("Δ Accuracy (relative)")
    plt.title(f"{title} at Distance = {distance}")
    plt.legend(title="Model", frameon=True, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out_path = FIG_DIR / "crowding" / f"{mode}_vs_strength.pdf"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved → {out_path}")
    
    
def plot_crowding_confidence_strength_effect(distance=20):

    set_thesis_style()

    df = load_analysis_data("crowding_results")
    df["foveation"] = df["foveation"].str.replace("-nosal", "-medium", regex=False)
    df = df[df["distance"] == distance].copy()

    # --- split groups ---
    def parse(row):
        name = row["foveation"]

        if "blur" in name:
            group = "blur"
        elif "cm" in name:
            group = "cm"
        else:
            group = "other"

        if "light" in name:
            strength = "light"
        elif "strong" in name:
            strength = "strong"
        elif group in ["blur", "cm"]:
            strength = "medium"
        else:
            strength = "constant"

        return pd.Series([group, strength])

    df[["group", "strength"]] = df.apply(parse, axis=1)

    # --- normalize to a ---
    df["xax_rel"] = df["xax_conf"] / df["a_conf"]

    # --- separate baseline + crop ---
    base_val = df[df["foveation"] == "base"]["xax_rel"].values[0]
    crop_val = df[df["foveation"] == "crop"]["xax_rel"].values[0]

    # --- keep only blur + cm ---
    df_plot = df[df["group"].isin(["blur", "cm"])].copy()

    # ordering
    strength_order = ["light", "medium", "strong"]
    df_plot["strength"] = pd.Categorical(
        df_plot["strength"],
        categories=strength_order,
        ordered=True
    )

    palette = get_foveation_palette()

    # ======================
    # PLOT
    # ======================
    fig, ax = plt.subplots(figsize=(6, 4))

    for group in ["blur", "cm"]:
        sub = df_plot[df_plot["group"] == group]

        ax.plot(
            sub["strength"],
            sub["xax_rel"],
            marker="o",
            linewidth=2,
            label=group.capitalize(),
            color=palette[group]
        )

    # --- reference lines ---
    ax.axhline(1.0, linestyle=":", color="gray", linewidth=1)

    ax.axhline(
        base_val,
        linestyle="--",
        color=palette["base"],
        linewidth=2,
        label="Base"
    )

    ax.axhline(
        crop_val,
        linestyle="--",
        color=palette["crop"],
        linewidth=2,
        label="Crop"
    )

    # --- labels ---
    ax.set_xlabel("Foveation Strength")
    ax.set_ylabel("Relative Confidence (xax / a)")
    ax.set_title(f"Confidence Drop under Crowding (All Predictions)", fontsize=12)

    # --- limits ---
    ymin = df_plot["xax_rel"].min() - 0.02
    ymax = 1.02
    ax.set_ylim(ymin, ymax)

    # --- legend ---
    ax.legend(
        frameon=True,
        ncol=2,
        fontsize=9
    )

    plt.tight_layout()

    out_path = FIG_DIR / "crowding" / f"confidence_strength_effect_d{distance}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved → {out_path}")
    
    
if __name__ == "__main__":
    print("-------------------------------------------------------------------------------")
    #plot_lr_sweep_knn_mean()
    #plot_linear_eval_foveated()
    #plot_delta_to_central_gaze()
    #plot_linear_eval_foveated_offline()
    #plot_delta_to_central_gaze_offline()
    
    #plot_inpainted_trend()
    #build_master_csv()
    #compute_background_stats()
    #compute_top_predictions()
    #compute_most_hallucinated_classes()
    
    #plot_ooc_delta_heatmap()
    #plot_color_std_object_only()
    #plot_object_or_ooc_trend("Object-Only") # Object-Only or OOC    
    #plot_object_or_ooc_trend("OOC")
    #plot_gap("ori", "ooc") # "ori", "obj", "ooc"
    #plot_gap("ori", "obj")
    #plot_gap("obj", "ooc")
    #export_ooc_background_summary()
    #plot_model_ooc_confidence()
    #plot_ooc_confidence_gaps(1) # 1,2,3
    #plot_ooc_confidence_gaps(2)
    #plot_ooc_confidence_gaps(3)
    #plot_ooc_confidence_dynamics()
    
    #plot_small_objects_accuracy()
    #compute_size_correlations()
    #plot_accuracy_vs_size()
    #plot_accuracy_vs_size_by_strength(method="blur") # blur or cm
    #plot_accuracy_vs_size_by_strength(method="cm")

    #plot_crowding_distance_trend(condition="ax", normalize=True) # ax or xax
    #plot_crowding_distance_trend(condition="xax", normalize=True) # ax or xax
    #plot_crowding_vs_strength(mode="a_xax") # a_ax, a_xax, ax_xax
    #plot_crowding_vs_strength(mode="a_ax")
    #plot_crowding_vs_strength(mode="ax_xax")
    #plot_crowding_intro()
    #plot_crowding_confidence_strength_effect()
    