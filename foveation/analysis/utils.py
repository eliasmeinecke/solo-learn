from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DATA_DIR = OUTPUT_DIR / "data"

DATASET_RENAME = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "imagenet": "ImageNet-1k 100%",
    "imagenet10pct": "ImageNet-1k 10%",
    "imagenet1pct": "ImageNet-1k 1%",
    "imagenet100": "ImageNet-100",
    "core50": "Core50",
    "toybox": "ToyBox",
    "gaze": "Foveated Imagenet"
}

DATASET_CATEGORIES = {
    # Hard category recognition
    "ImageNet-1k 100%": "Hard category recognition",
    "ImageNet-100": "Hard category recognition",
    "ImageNet-1k 10%": "Hard category recognition",
    "ImageNet-1k 1%": "Hard category recognition",
    "CIFAR-100": "Hard category recognition",

    # Easy category recognition
    "CIFAR-10": "Easy category recognition",

    # Fine-grained recognition
    "DTD": "Fine-grained recognition",
    "FGVCAircraft": "Fine-grained recognition",
    "Flowers102": "Fine-grained recognition",
    "OxfordIIITPet": "Fine-grained recognition",
    "StanfordCars": "Fine-grained recognition",

    # Instance recognition
    "COIL100": "Instance recognition",
    "Core50": "Instance recognition",
    "ToyBox": "Instance recognition"
}

DATASET_ORDER = [
    "ImageNet-1k 100%",
    "ImageNet-1k 10%",
    "ImageNet-1k 1%",
    "ImageNet-100",
    "CIFAR-100",
    "CIFAR-10",
    "DTD",
    "FGVCAircraft",
    "Flowers102",
    "OxfordIIITPet",
    "StanfordCars",
    "COIL100",
    "Core50",
    "ToyBox",
    "Foveated Imagenet"
]

FOVEATION_ORDER = [
    "base",
    "crop",
    "blur",
    "cm"
]

DATASET_RENAME_LATEX = {
    "ImageNet-1k 100%": "ImageNet-1k 100\\%",
    "ImageNet-1k 10%": "ImageNet-1k 10\\%",
    "ImageNet-1k 1%": "ImageNet-1k 1\\%",
}


def save_data(df: pd.DataFrame, name: str):
    path = DATA_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"Saved data → {path}")
    

def load_analysis_data(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find data file: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {name}.csv ({len(df)} rows)")
    return df


def set_thesis_style():
    plt.style.use("default")

    plt.rcParams.update({
        "figure.figsize": (6, 4),
        "figure.dpi": 100,

        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",

        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,

        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        "lines.linewidth": 2,
        "lines.markersize": 6,
    })


def extract_foveation_type(df):
    def parse(name):
        if "blur-nosal" in name:
            return "blur"
        if "cm-nosal" in name:
            return "cm"
        # ignore saliency variants
        if "blur-sal" in name or "cm-sal" in name:
            return None
        if "crop" in name:
            return "crop"
        if "base" in name:
            return "base"
        return None
    df["foveation"] = df["name"].apply(parse)
    df = df[df["foveation"].notna()]
    return df


def extract_exact_foveation_type(df):
    def parse(name):
        if "blur-light" in name:
            return "blur-light"
        if "blur-nosal" in name:
            return "blur"
        if "blur-strong" in name:
            return "blur-strong"
        if "cm-light" in name:
            return "cm-light"
        if "cm-nosal" in name:
            return "cm"
        if "cm-strong" in name:
            return "cm-strong"
        if "crop" in name:
            return "crop"
        if "base" in name:
            return "base"
        return None
    df["foveation"] = df["name"].apply(parse)
    df = df[df["foveation"].notna()]
    return df

def add_dataset_category(df):
    df["category"] = df["dataset"].map(DATASET_CATEGORIES)
    return df


def clean_dataset_names(df):
    df["dataset"] = df["dataset"].map(DATASET_RENAME).fillna(df["dataset"])
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    return df


def add_foveation_order(df):
    df["foveation"] = pd.Categorical(df["foveation"], categories=FOVEATION_ORDER, ordered=True)
    return df


def fix_dataset_order_in_table(table):
    table = table.reset_index()

    table["dataset"] = pd.Categorical(
        table["dataset"],
        categories=DATASET_ORDER,
        ordered=True
    )

    table = table.sort_values("dataset")

    table = table.set_index(["category", "dataset"])
    return table

def compute_category_average(df):
    avg = (
        df.groupby(["category", "foveation"])["linear_acc1_best"]
        .mean()
        .reset_index()
    )
    return avg


def build_linear_eval_table():

    df = load_analysis_data("linear_eval_processed")

    table = df.pivot_table(
        index=["category", "dataset"],
        columns="foveation",
        values="linear_acc1_best"
    )
    
    table = fix_dataset_order_in_table(table)

    # rename datasets for LaTeX
    new_index = []

    for category, dataset in table.index:

        dataset_latex = DATASET_RENAME_LATEX.get(dataset, dataset)

        new_index.append((category, dataset_latex))

    table.index = pd.MultiIndex.from_tuples(
        new_index,
        names=table.index.names
    )

    return table


def highlight_best(row):

    max_val = row.max()

    return [
        f"\\textbf{{{v:.2f}}}" if v == max_val else f"{v:.2f}"
        for v in row
    ]


def add_category_averages(table):

    blocks = []

    for category in table.index.get_level_values(0).unique():

        subset = table.loc[category]

        # Average berechnen
        avg = subset.mean()

        avg_df = pd.DataFrame([avg])
        avg_df.index = pd.MultiIndex.from_tuples(
            [(category, "Average")],
            names=table.index.names
        )

        # Dataset rows wieder MultiIndex geben
        subset.index = pd.MultiIndex.from_product(
            [[category], subset.index],
            names=table.index.names
        )

        block = pd.concat([subset, avg_df])

        blocks.append(block)

    table = pd.concat(blocks)

    return table