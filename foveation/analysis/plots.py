import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np

from .utils import (
    load_analysis_data, 
    set_thesis_style, 
    build_linear_eval_table, 
    highlight_best, 
    add_category_averages,
    fix_dataset_order_in_table,
    compute_confidence_gap,
    parse_model,
    get_group,
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


def plot_knn_mean(metric="top1"):

    df = load_analysis_data("pretrain_processed")

    cols = [c for c in df.columns if f"knn_{metric}_k" in c]

    df[f"mean_{metric}"] = df[cols].mean(axis=1)

    df_sorted = df.set_index("foveation").loc[FOVEATION_ORDER].reset_index()

    plt.figure(figsize=(8,5))

    sns.barplot(
        data=df_sorted,
        x="foveation",
        y=f"mean_{metric}",
        hue="foveation",
        order=FOVEATION_ORDER,
        palette = sns.color_palette("colorblind", len(FOVEATION_ORDER))
    )

    values = df_sorted[f"mean_{metric}"]

    plt.ylim(values.min() - 0.3, values.max() + 0.3)

    for i, val in enumerate(values):
        plt.text(
            i,
            val + 0.05,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.ylabel(f"Mean kNN {metric.upper()} Accuracy (%)")
    plt.xlabel("Foveation Method")

    plt.title(f"Pretraining Comparison (Mean kNN {metric.upper()} over K)")

    plt.xticks(rotation=25)

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    out_path = FIG_DIR / "central_knn" /f"knn_mean_{metric}.pdf"

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved figure → {out_path}")

    
def plot_linear_eval_heatmap():

    df = load_analysis_data("linear_eval_processed")

    table = df.pivot_table(
        index=["category", "dataset"],
        columns="foveation",
        values="linear_acc1_best"
    )
    
    table = fix_dataset_order_in_table(table)

    table = table[FOVEATION_ORDER]

    datasets = table.index.get_level_values("dataset")
    categories = table.index.get_level_values("category")

    values = table.values
    
    annot = values.astype(str)
    for i in range(values.shape[0]):
        best = np.argmax(values[i])
        
        for j in range(values.shape[1]):
            if j == best:
                annot[i, j] = f"$\\bf{{{values[i,j]:.1f}}}$"
            else:
                annot[i, j] = f"{values[i,j]:.1f}"
    
    plt.figure(figsize=(9,7))

    ax = sns.heatmap(
        values,
        annot=annot,
        fmt="",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"label": "Linear-Probe Accuracy (%)"},
        annot_kws={"size":10}
    )

    ax.set_xticklabels(
        ["base","crop","blur","cm"],
        rotation=0
    )
    ax.set_yticklabels(datasets, rotation=0)

    ax.set_xlabel("Foveation Method")
    ax.set_ylabel("Dataset")
    ax.set_title("Linear Probe Accuracy Across Datasets")
        
    category_changes = []

    prev = categories[0]

    for i, cat in enumerate(categories):
        if cat != prev:
            category_changes.append(i)
            prev = cat

    for y in category_changes:
        ax.axhline(y, color="white", lw=4)

    plt.tight_layout()
    out_path = FIG_DIR / "central_linear_eval" / "linear_eval_heatmap.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved figure → {out_path}")
    

def plot_delta_to_baseline():

    df = load_analysis_data("linear_eval_processed")

    table = df.pivot_table(
        index=["category", "dataset"],
        columns="foveation",
        values="linear_acc1_best"
    )

    table = fix_dataset_order_in_table(table)

    table = table[FOVEATION_ORDER]

    baseline = table["base"]

    delta = table.subtract(baseline, axis=0)

    delta = delta.drop(columns=["base"])

    datasets = delta.index.get_level_values("dataset")
    values = delta.values

    plt.figure(figsize=(9,7))

    ax = sns.heatmap(
        values,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Δ Accuracy"}
    )

    ax.set_xticklabels(
        ["crop","blur","cm"],
        rotation=0
    )
    ax.set_yticklabels(datasets, rotation=0)

    ax.set_xlabel("Foveation Method")
    ax.set_ylabel("Dataset")
    ax.set_title("Improvement over Baseline")

    plt.tight_layout()

    out_path = FIG_DIR / "central_linear_eval" / "linear_eval_delta.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved figure → {out_path}")
    
    
def plot_category_deltas():
    
    df = load_analysis_data("linear_eval_processed")

    avg = (
        df.groupby(["category", "foveation"])["linear_acc1_best"]
        .mean()
        .reset_index()
    )

    baseline = avg[avg["foveation"] == "base"][
        ["category", "linear_acc1_best"]
    ].rename(columns={"linear_acc1_best": "baseline_acc"})

    avg = avg.merge(baseline, on="category")

    avg["delta"] = avg["linear_acc1_best"] - avg["baseline_acc"]

    plt.figure(figsize=(8,5))

    ax = sns.barplot(
        data=avg,
        x="category",
        y="delta",
        hue="foveation",
        hue_order=FOVEATION_ORDER,
        palette = sns.color_palette("colorblind", len(FOVEATION_ORDER))
    )

    ax.axhline(0, color="black", lw=1)

    ax.set_ylabel("Δ Accuracy vs Baseline (%)")
    ax.set_xlabel("")
    ax.set_title("Average Improvement per Category")

    plt.xticks(rotation=20)

    plt.tight_layout()

    out_path = FIG_DIR / "central_linear_eval" / "linear_eval_category_delta.pdf"
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
    
    
def plot_ooc_delta_heatmap():

    set_thesis_style()

    df = load_analysis_data("ooc_results")

    df["model"] = df["model"].str.replace("-nosal", "", regex=False)

    # --- baseline ---
    baseline = df[df["model"] == "base"].iloc[0]

    # --- compute delta ---
    df_delta = df.copy()
    for col in OOC_DATASET_ORDER:
        df_delta[col] = df[col] - baseline[col]

    df_delta = df_delta.set_index("model")[OOC_DATASET_ORDER]

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
    plt.title("OOC Improvement over Baseline", fontsize=12, pad=60)

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
    plt.title("Performance on Background-Only (Inpainted) Images")

    plt.legend(frameon=True)
    plt.tight_layout()

    out_path = FIG_DIR / "ooc_eval" / "inpainted_trend.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved → {out_path}")
    

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
    df["base_model"] = df["base_model"].str.replace("-nosal", "", regex=False)

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

    df_std["group"] = df_std["base_model"].apply(get_group)

    # --- plot ---
    plt.figure(figsize=(8, 4))

    sns.barplot(
        data=df_std,
        x="base_model",
        y="std",
        hue="group",
        palette=FOVEATION_PALETTE,
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
    
    
def plot_object_trend():
    
    set_thesis_style()

    df = load_analysis_data("ooc_results")

    # remove variants
    df = df[~df["model"].str.contains("__")]

    df["model"] = df["model"].str.replace("-nosal", "", regex=False)

    df = df[["model", "object"]].copy()

    df[["type", "strength"]] = df["model"].apply(
        lambda x: pd.Series(parse_model(x))
    )

    # --- only blur + cm lines ---
    df_lines = df[df["type"].isin(["blur", "cm"])]

    # --- scale ---
    df_plot = df_lines.copy()
    df_plot["object"] *= 100

    # --- plot ---
    plt.figure(figsize=(6, 4))

    sns.lineplot(
        data=df_plot,
        x="strength",
        y="object",
        hue="type",
        palette=FOVEATION_PALETTE,
        marker="o",
        errorbar=None
    )

    # --- constant lines ---
    base_val = df[df["model"] == "base"]["object"].values[0] * 100
    crop_val = df[df["model"] == "crop"]["object"].values[0] * 100

    plt.axhline(base_val, linestyle="--", color=FOVEATION_PALETTE["base"], label="base")
    plt.axhline(crop_val, linestyle="--", color=FOVEATION_PALETTE["crop"], label="crop")

    # --- y-limits ---
    y_min = min(df_plot["object"].min(), base_val, crop_val)
    y_max = max(df_plot["object"].max(), base_val, crop_val)

    plt.ylim(y_min * 0.95, y_max * 1.05)

    # --- x-axis ---
    plt.xticks([2, 3, 4], ["light", "medium", "strong"])

    # --- labels ---
    plt.xlabel("Foveation Strength")
    plt.ylabel("Accuracy (%)")
    plt.title("Performance on Object-Only Images")

    # --- legend ---
    plt.legend(frameon=True)

    plt.tight_layout()

    out_path = FIG_DIR / "ooc_eval" / "object_trend.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved → {out_path}")
    
    
def plot_gap(dataset_a, dataset_b):
    
    set_thesis_style()
    
    dataset_map = {"ori": "original", "inp": "inpainted", "obj": "object", "ooc": "ooc"}
    dataset_labels = {"ori": "Original", "inp": "Background-Only", "obj": "Object-Only", "ooc": "Out-of-Context"}
    
    if dataset_a not in dataset_map or dataset_b not in dataset_map:
        raise ValueError("Invalid dataset keys")

    col_a = dataset_map[dataset_a]
    col_b = dataset_map[dataset_b]

    label_a = dataset_labels[dataset_a]
    label_b = dataset_labels[dataset_b]
    df = load_analysis_data("ooc_results")

    # --- remove variants ---
    df = df[~df["model"].str.contains("__")]

    df["model"] = df["model"].str.replace("-nosal", "", regex=False)

    # --- compute gap ---
    df["gap"] = df[col_a] - df[col_b]

    df = df[["model", "gap"]].copy()

    df[["type", "strength"]] = df["model"].apply(
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
    base_gap = df[df["model"] == "base"]["gap"].values[0] * 100
    crop_gap = df[df["model"] == "crop"]["gap"].values[0] * 100

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
    plt.ylabel(f"{label_a} − {label_b} Accuracy (%)")
    plt.title(f"{label_a} vs {label_b} Gap")

    # --- legend ---
    plt.legend(frameon=True)

    plt.tight_layout()

    # --- filename ---
    out_path = FIG_DIR / "ooc_eval" / f"gap_{dataset_a}_vs_{dataset_b}.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved → {out_path}")
    
    
def plot_model_ooc_confidence():
    
    set_thesis_style()

    models = ["base", "cm-strong"]

    dfs = []

    for model in models:
        model_path = DATA_DIR / "ooc_per_sample" / model

        for d in OOC_DATASET_ORDER:
            path = model_path / f"{model}_{d}.csv"

            df = pd.read_csv(path)

            df["dataset"] = d
            df["model"] = model

            dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # --- clean ---
    df_all["correct"] = df_all["correct"].astype(bool)
    df_all["model"] = df_all["model"].str.replace("-nosal", "", regex=False)

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

    plt.suptitle(
        "Confidence Distributions: Baseline vs CM-Strong",
        fontsize=13,
        y=0.98
    )
    
    g._legend.set_title("Prediction")
    for t, l in zip(g._legend.texts, ["Incorrect", "Correct"]):
        t.set_text(l)

    out_path = FIG_DIR / "ooc_confidence" / "model_confidence_comparison.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved → {out_path}")
    
    
def plot_ooc_confidence_gap(mode="correct"):

    set_thesis_style()

    models = [
        "base", "crop",
        "blur-nosal", "blur-light", "blur-strong",
        "cm-nosal", "cm-light", "cm-strong"
    ]

    df = compute_confidence_gap(models, mode=mode)

    # --- clean names ---
    df["model"] = df["model"].str.replace("-nosal", "", regex=False)

    df[["type", "strength"]] = df["model"].apply(
        lambda x: pd.Series(parse_model(x))
    )

    df_plot = df[df["type"].isin(["blur", "cm"])].copy()
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

    # --- baselines ---
    base_gap = df[df["model"] == "base"]["gap"].values[0] * 100
    crop_gap = df[df["model"] == "crop"]["gap"].values[0] * 100

    plt.axhline(base_gap, linestyle="--", color=FOVEATION_PALETTE["base"], linewidth=2, label="base")
    plt.axhline(crop_gap, linestyle="--", color=FOVEATION_PALETTE["crop"], linewidth=2, label="crop")

    plt.axhline(0, linestyle=":", color="gray", linewidth=1)
    
    # --- y-limits ---
    y_min = min(df_plot["gap"].min(), base_gap, crop_gap)
    y_max = max(df_plot["gap"].max(), base_gap, crop_gap)

    plt.ylim(y_min - 0.5, y_max * 1.05)

    # --- axis ---
    plt.xticks([2, 3, 4], ["light", "medium", "strong"])

    plt.xlabel("Foveation Strength")
    plt.ylabel("Confidence Gap (%)")

    title_map = {
        "correct": "OOC Confidence Drop (Correct Predictions)",
        "incorrect": "OOC Confidence Drop (Incorrect Predictions)",
        "all": "OOC Confidence Drop (All Predictions)"
    }
    plt.title(title_map[mode])

    plt.legend(frameon=True)

    plt.tight_layout()

    out_path = FIG_DIR / "ooc_confidence" / f"ooc_confidence_gap_{mode}.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved → {out_path}")
    
    
def plot_small_objects_accuracy():
    
    set_thesis_style()

    SMALL_THRESH = 0.05

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

    print(f"Total samples (small objects): {len(df_small)}")

    # --- compute accuracy per model ---
    df_acc = (
        df_small
        .groupby("model")["correct"]
        .mean()
        .reset_index()
    )

    # scale to %
    df_acc["accuracy"] = df_acc["correct"] * 100

    # clean names
    df_acc["model"] = df_acc["model"].str.replace("-nosal", "", regex=False)

    # enforce order
    df_acc["model"] = pd.Categorical(
        df_acc["model"],
        categories=EXACT_FOVEATION_ORDER,
        ordered=True
    )
    df_acc = df_acc.sort_values("model")

    df_acc["group"] = df_acc["model"].apply(get_group)
    
    # --- plot ---
    plt.figure(figsize=(7, 4))

    sns.barplot(
        data=df_acc,
        x="model",
        y="accuracy",
        hue="group",
        palette=FOVEATION_PALETTE
    )

    # value labels
    for i, v in enumerate(df_acc["accuracy"]):
        plt.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)

    plt.ylabel("Accuracy (%)")
    plt.xlabel("Model")
    plt.title(f"Performance on Small Objects (mask_area ≤ {SMALL_THRESH})")
    
    plt.xticks(rotation=0)
    plt.ylim(0, df_acc["accuracy"].max() + 5)
    
    plt.legend().remove()

    plt.tight_layout()

    out_path = FIG_DIR / "size_analysis" / "small_objects_accuracy.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved → {out_path}")
    
    
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
    n_bins = 12
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

    plt.xlabel("Object Size (mask_area)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Object Size")

    plt.tight_layout()

    out_path = FIG_DIR / "size_analysis" / "accuracy_vs_size.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved → {out_path}")
    
    
def plot_accuracy_vs_size_by_strength(method="blur"):
    
    set_thesis_style()

    models = [
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
    df_all = df_all[df_all["type"] == method]

    # --- map strength labels ---
    strength_map = {
        2: "light",
        3: "medium",
        4: "strong"
    }
    df_all["strength_label"] = df_all["strength"].map(strength_map)

    # --- binning ---
    n_bins = 12
    df_all["size_bin"] = pd.qcut(df_all["mask_area"], q=n_bins, duplicates="drop")
    df_all["bin_center"] = df_all["size_bin"].apply(lambda x: x.mid)

    # --- aggregate ---
    df_plot = (
        df_all
        .groupby(["strength_label", "bin_center"])["correct"]
        .mean()
        .reset_index()
    )

    df_plot["accuracy"] = df_plot["correct"] * 100

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
        y="accuracy",
        hue="strength_label",
        palette=palette,
        marker="o"
    )

    plt.xlabel("Object Size (mask_area)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{method.upper()} – Accuracy vs Object Size")

    plt.legend(title="Strength", frameon=False)

    plt.tight_layout()

    out_path = FIG_DIR / "size_analysis" / f"{method}_size_vs_accuracy.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved → {out_path}")
        
    
def export_linear_eval_latex():

    table = build_linear_eval_table()
    table = add_category_averages(table)
    
    styled_rows = []
    for idx, row in table.iterrows():
        styled_rows.append(highlight_best(row))

    styled = pd.DataFrame(
        styled_rows,
        index=table.index,
        columns=table.columns
    )

    styled = styled.reset_index()

    styled = styled.rename(columns={
        "dataset": "Dataset",
        "category": "Category"
    })

    # ---- generate latex ----
    latex = styled.to_latex(
        index=False,
        escape=False
    )

    # ---- insert category separators ----
    lines = latex.split("\n")
    new_lines = []

    prev_category = None

    for line in lines:

        if "&" in line and not line.startswith("\\"):

            category = line.split("&")[0].strip()

            if prev_category is not None and category != prev_category and category != "":
                new_lines.append("\\midrule")

            prev_category = category if category != "" else prev_category

        new_lines.append(line)

    latex = "\n".join(new_lines)

    out_path = FIG_DIR / "central_linear_eval" / "linear_eval_table.tex"

    with open(out_path, "w") as f:
        f.write(latex)

    print(f"Saved LaTeX table → {out_path}")
    
    
if __name__ == "__main__":
    print("-------------------------------------------------------------------------------")
    #plot_linear_eval_foveated()
    #plot_delta_to_central_gaze()
    #plot_ooc_delta_heatmap()
    #plot_inpainted_trend()
    #plot_color_std_object_only()
    #plot_object_trend()
    #plot_gap("obj", "ooc") # "ori", "inp", "obj", "ooc"
    #plot_model_ooc_confidence()
    #plot_ooc_confidence_gap(mode="all") # modes: correct, incorrect, all 
    #plot_small_objects_accuracy()
    #plot_accuracy_vs_size()
    #plot_accuracy_vs_size_by_strength(method="blur") # blur or cm
    #plot_accuracy_vs_size_by_strength(method="cm")
    
    # plot crowding results??
    