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
    FOVEATION_ORDER,
    EXACT_FOVEATION_ORDER,
    FOVEATION_PALETTE,
    OOC_DATASET_ORDER
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
    
    df = load_analysis_data("gaze_linear_eval_processed")
    df = df[df["dataset"] == "Foveated Imagenet"]

    df["foveation"] = pd.Categorical(df["foveation"], categories=EXACT_FOVEATION_ORDER, ordered=True)
    df = df.sort_values("foveation")

    # --- baseline ---
    baseline = df[df["foveation"] == "base"]["linear_acc1_best"].iloc[0]

    # --- delta ---
    df["delta"] = df["linear_acc1_best"] - baseline

    # --- nicer grouping colors ---
    def get_group(f):
        if "blur" in f:
            return "blur"
        elif "cm" in f:
            return "cm"
        elif f == "crop":
            return "crop"
        else:
            return "base"

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

    # labels
    plt.ylabel("Δ Top-1 Accuracy vs Baseline (%)")
    plt.xlabel("Foveation Type")
    plt.title(f"Foveated ImageNet Linear Eval Improvement (Baseline = {baseline:.1f}%)")

    # annotate values
    for i, v in enumerate(df["delta"]):
        plt.text(i, v + 0.15, f"{v:+.1f}", ha="center", fontsize=9)

    plt.xticks(rotation=0)

    # legend cleanup
    plt.legend(title="Method", frameon=False)

    plt.tight_layout()

    out_path = FIG_DIR / "gaze_linear_eval" / "gaze_linear_eval_delta.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved figure → {out_path}")
    
    
def plot_delta_to_central_gaze():

    # load both datasets
    df_gaze = load_analysis_data("gaze_linear_eval_processed")
    df_imnet = load_analysis_data("linear_eval_processed")

    # filter
    df_gaze = df_gaze[df_gaze["dataset"] == "Foveated Imagenet"]
    df_imnet = df_imnet[df_imnet["dataset"] == "ImageNet-1k 100%"]

    # keep only comparable methods
    methods = ["base", "crop", "blur", "cm"]

    df_gaze = df_gaze[df_gaze["foveation"].isin(methods)]
    df_imnet = df_imnet[df_imnet["foveation"].isin(methods)]

    # merge
    df = pd.merge(
        df_gaze[["foveation", "linear_acc1_best"]],
        df_imnet[["foveation", "linear_acc1_best"]],
        on="foveation",
        suffixes=("_gaze", "_imagenet")
    )

    # compute delta
    df["delta"] = df["linear_acc1_best_gaze"] - df["linear_acc1_best_imagenet"]

    # order
    df["foveation"] = pd.Categorical(df["foveation"], categories=FOVEATION_ORDER, ordered=True)
    df = df.sort_values("foveation")

    # plot
    plt.figure(figsize=(7, 4))

    sns.barplot(
        data=df,
        x="foveation",
        y="delta",
        hue="foveation",
        palette=FOVEATION_PALETTE,
        legend=False
    )

    plt.ylabel("Δ Accuracy (Object – Central Gaze) (%)")
    plt.xlabel("Foveation Type")
    plt.title("Improvement Object vs Central Gaze (Foveated ImageNet)")

    # annotate
    for i, v in enumerate(df["delta"]):
        plt.text(i, v + 0.15, f"{v:+.1f}", ha="center", fontsize=9)

    plt.xticks(rotation=0)

    # nicer y-limits
    plt.ylim(top=df["delta"].max() + 0.5)

    plt.tight_layout()

    out_path = FIG_DIR / "gaze_linear_eval" / "central_gaze_delta.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved figure → {out_path}")
    
    
def plot_ooc_heatmap():

    df = load_analysis_data("ooc_results")

    # clean names
    df["model"] = df["model"].str.replace("-nosal", "", regex=False)

    df = df.set_index("model")[OOC_DATASET_ORDER]
    df = df.loc[EXACT_FOVEATION_ORDER]
    
    df_plot = df.copy()
    df_plot[OOC_DATASET_ORDER] = df_plot[OOC_DATASET_ORDER] * 100

    plt.figure(figsize=(7, 4))

    sns.heatmap(
        df_plot,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        cbar_kws={"label": "Accuracy"}
    )

    plt.title("OOC Evaluation Heatmap")
    plt.xlabel("")
    plt.ylabel("Model")

    plt.tight_layout()

    out_path = FIG_DIR / "ooc_eval" / "ooc_heatmap.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved → {out_path}")
    
    
def plot_ooc_delta_heatmap():

    df = load_analysis_data("ooc_results")

    df["model"] = df["model"].str.replace("-nosal", "", regex=False)

    baseline = df[df["model"] == "base"].iloc[0]

    df_delta = df.copy()
    for col in OOC_DATASET_ORDER:
        df_delta[col] = df[col] - baseline[col]

    df_delta = df_delta.set_index("model")[OOC_DATASET_ORDER]
    df_delta = df_delta.loc[EXACT_FOVEATION_ORDER]
    
    df_plot = df_delta.copy()
    df_plot[OOC_DATASET_ORDER] = df_plot[OOC_DATASET_ORDER] * 100

    plt.figure(figsize=(7, 4))

    sns.heatmap(
        df_plot,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "Δ Accuracy vs Base"}
    )

    plt.title("Improvement over Baseline")
    plt.tight_layout()

    out_path = FIG_DIR / "ooc_eval" / "ooc_delta_heatmap.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved → {out_path}")
    
    
def plot_inpainted_trend():

    df = load_analysis_data("ooc_results")

    df["model"] = df["model"].str.replace("-nosal", "", regex=False)

    df = df[["model", "inpainted"]].copy()

    # --- define groups ---
    def parse_model(m):
        if m == "base":
            return "base", 0
        if m == "crop":
            return "crop", 1
        if "blur" in m:
            if "light" in m:
                return "blur", 2
            elif "strong" in m:
                return "blur", 4
            else:
                return "blur", 3
        if "cm" in m:
            if "light" in m:
                return "cm", 2
            elif "strong" in m:
                return "cm", 4
            else:
                return "cm", 3
        return "other", -1

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
        marker="o"
    )

    # baseline horizontal line
    base_val = df[df["model"] == "base"]["inpainted"].values[0] * 100
    plt.axhline(base_val, linestyle="--", color=FOVEATION_PALETTE["base"], label="base")
    base_val = df[df["model"] == "crop"]["inpainted"].values[0] * 100
    plt.axhline(base_val, linestyle="--", color=FOVEATION_PALETTE["crop"], label="crop")

    plt.xticks(
        [2, 3, 4],
        ["light", "medium", "strong"]
    )

    plt.xlabel("Foveation Strength")
    plt.ylabel("Accuracy (%)")
    plt.title("Performance on Background-Only (Inpainted) Images")

    plt.legend()
    plt.tight_layout()

    out_path = FIG_DIR / "ooc_eval" / "inpainted_trend.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved → {out_path}")
    
    
def plot_object_trend():

    df = load_analysis_data("ooc_results")

    df["model"] = df["model"].str.replace("-nosal", "", regex=False)

    df = df[["model", "object"]].copy()

    # --- define groups ---
    def parse_model(m):
        if m == "base":
            return "base", 0
        if m == "crop":
            return "crop", 1
        if "blur" in m:
            if "light" in m:
                return "blur", 2
            elif "strong" in m:
                return "blur", 4
            else:
                return "blur", 3
        if "cm" in m:
            if "light" in m:
                return "cm", 2
            elif "strong" in m:
                return "cm", 4
            else:
                return "cm", 3
        return "other", -1

    df[["type", "strength"]] = df["model"].apply(
        lambda x: pd.Series(parse_model(x))
    )

    # only blur + cm lines
    df_lines = df[df["type"].isin(["blur", "cm"])]

    # scale
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
        marker="o"
    )

    # --- reference lines ---
    base_val = df[df["model"] == "base"]["object"].values[0] * 100
    crop_val = df[df["model"] == "crop"]["object"].values[0] * 100

    plt.axhline(base_val, linestyle="--", color=FOVEATION_PALETTE["base"], label="base")
    plt.axhline(crop_val, linestyle="--", color=FOVEATION_PALETTE["crop"], label="crop")

    # --- x-axis ---
    plt.xticks(
        [2, 3, 4],
        ["light", "medium", "strong"]
    )

    plt.xlabel("Foveation Strength")
    plt.ylabel("Accuracy (%)")
    plt.title("Performance on Object-Only Images")

    plt.legend()
    plt.tight_layout()

    out_path = FIG_DIR / "ooc_eval" / "object_trend.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved → {out_path}")
    
    
def plot_ooc_gap():

    df = load_analysis_data("ooc_results")

    df["model"] = df["model"].str.replace("-nosal", "", regex=False)

    # --- compute gap ---
    df["gap"] = df["object"] - df["ooc"]

    df = df[["model", "gap"]].copy()

    # --- parse ---
    def parse_model(m):
        if m == "base":
            return "base", 0
        if m == "crop":
            return "crop", 1
        if "blur" in m:
            if "light" in m:
                return "blur", 2
            elif "strong" in m:
                return "blur", 4
            else:
                return "blur", 3
        if "cm" in m:
            if "light" in m:
                return "cm", 2
            elif "strong" in m:
                return "cm", 4
            else:
                return "cm", 3
        return "other", -1

    df[["type", "strength"]] = df["model"].apply(
        lambda x: pd.Series(parse_model(x))
    )

    # only lines
    df_lines = df[df["type"].isin(["blur", "cm"])]

    # scale to %
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
        marker="o"
    )

    # --- baseline & crop reference ---
    base_gap = df[df["model"] == "base"]["gap"].values[0] * 100
    crop_gap = df[df["model"] == "crop"]["gap"].values[0] * 100

    plt.axhline(base_gap, linestyle="--", color=FOVEATION_PALETTE["base"], label="base")
    plt.axhline(crop_gap, linestyle="--", color=FOVEATION_PALETTE["crop"], label="crop")

    # --- x-axis ---
    plt.xticks(
        [2, 3, 4],
        ["light", "medium", "strong"]
    )

    plt.xlabel("Foveation Strength")
    plt.ylabel("Object − OOC Accuracy (%)")
    plt.title("Context Sensitivity (Object vs OOC Gap)")

    plt.legend()
    plt.tight_layout()

    out_path = FIG_DIR / "ooc_eval" / "ooc_gap.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved → {out_path}")
    
    
def plot_model_confidence():

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

    # --- plot ---
    g = sns.displot(
        data=df_all,
        x="conf_top1",
        hue="correct",
        col="dataset",
        row="model",
        kind="kde",
        fill=True,
        common_norm=False,
        height=3,
        aspect=1
    )

    g.set_axis_labels("Confidence", "Density")
    g.set_titles("{row_name} | {col_name}")

    # fix limits
    for ax in g.axes.flat:
        ax.set_xlim(0, 1)

    plt.suptitle("Confidence Distribution: Baseline vs CM-Strong", y=1.02)

    out_path = FIG_DIR / "ooc_confidence" / "model_confidence_comparison.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved → {out_path}")
    
    
def plot_ooc_confidence_gap():

    models = [
        "base", "crop",
        "blur-nosal", "blur-light", "blur-strong",
        "cm-nosal", "cm-light", "cm-strong"
    ]

    results = []

    for model in models:

        model_path = DATA_DIR / "ooc_per_sample" / model

        dfs = {}

        for d in ["object", "ooc"]:
            path = model_path / f"{model}_{d}.csv"
            df = pd.read_csv(path)

            df = df[df["correct"] == 1]  # only looking at correct predictions here

            dfs[d] = df

        # mean confidence
        conf_object = dfs["object"]["conf_top1"].mean()
        conf_ooc = dfs["ooc"]["conf_top1"].mean()

        gap = conf_object - conf_ooc

        results.append({
            "model": model,
            "gap": gap
        })

    df = pd.DataFrame(results)

    # --- clean names ---
    df["model"] = df["model"].str.replace("-nosal", "", regex=False)

    # --- parse ---
    def parse_model(m):
        if m == "base":
            return "base", 0
        if m == "crop":
            return "crop", 1
        if "blur" in m:
            if "light" in m:
                return "blur", 2
            elif "strong" in m:
                return "blur", 4
            else:
                return "blur", 3
        if "cm" in m:
            if "light" in m:
                return "cm", 2
            elif "strong" in m:
                return "cm", 4
            else:
                return "cm", 3
        return "other", -1

    df[["type", "strength"]] = df["model"].apply(
        lambda x: pd.Series(parse_model(x))
    )

    df_plot = df[df["type"].isin(["blur", "cm"])].copy()
    df_plot["gap"] *= 100  # %

    # --- plot ---
    plt.figure(figsize=(6, 4))

    sns.lineplot(
        data=df_plot,
        x="strength",
        y="gap",
        hue="type",
        palette=FOVEATION_PALETTE,
        marker="o"
    )

    # --- reference lines ---
    base_gap = df[df["model"] == "base"]["gap"].values[0] * 100
    crop_gap = df[df["model"] == "crop"]["gap"].values[0] * 100

    plt.axhline(base_gap, linestyle="--", color=FOVEATION_PALETTE["base"], label="base")
    plt.axhline(crop_gap, linestyle="--", color=FOVEATION_PALETTE["crop"], label="crop")

    plt.xticks(
        [2, 3, 4],
        ["light", "medium", "strong"]
    )

    plt.xlabel("Foveation Strength")
    plt.ylabel("Confidence Drop (%)")
    plt.title("Confidence Drop: Object → OOC")

    plt.legend()
    plt.tight_layout()

    out_path = FIG_DIR / "ooc_confidence" / "ooc_confidence_gap.pdf"
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
    #plot_ooc_heatmap()
    #plot_ooc_delta_heatmap()
    #plot_inpainted_trend()
    #plot_object_trend()
    #plot_ooc_gap()
    #plot_model_confidence()
    plot_ooc_confidence_gap()
    