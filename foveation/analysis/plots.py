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
    FOVEATION_ORDER
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
    
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
    
    #export_linear_eval_latex