from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

EXACT_FOVEATION_ORDER = [
        "base",
        "crop",
        "blur-light",
        "blur",
        "blur-strong",
        "cm-light",
        "cm",
        "cm-strong",
]

FOVEATION_PALETTE = {
    "base": "#4D4D4D",     
    "crop": "#E69F00",     
    "blur": "#4C9F70",     
    "cm": "#5E3C99",     
}

OOC_DATASET_ORDER = ["original", "inpainted", "object", "ooc"]

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
    # --- base style ---
    plt.style.use("default")
    sns.set_theme(style="whitegrid")  # IMPORTANT for seaborn consistency

    plt.rcParams.update({

        # --- figure ---
        "figure.figsize": (6, 4),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",

        # --- axes ---
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,   # grid behind bars/lines

        # --- grid ---
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.color": "#BBBBBB",

        # --- fonts ---
        "font.size": 11,
        "font.family": "sans-serif",

        "axes.titlesize": 13,
        "axes.titleweight": "bold",

        "axes.labelsize": 11,

        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # --- ticks ---
        "xtick.direction": "out",
        "ytick.direction": "out",

        # --- legend ---
        "legend.fontsize": 10,
        "legend.frameon": False,

        # --- lines ---
        "lines.linewidth": 2,
        "lines.markersize": 6,

        # --- barplots ---
        "patch.edgecolor": "none",   # removes ugly borders
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


def compute_confidence_gap(models, mode="correct"):
    
    results = []

    for model in models:

        model_path = DATA_DIR / "ooc_per_sample" / model
        dfs = {}

        for d in ["object", "ooc"]:
            path = model_path / f"{model}_{d}.csv"
            df = pd.read_csv(path)

            if mode == "correct":
                df = df[df["correct"] == 1]
            elif mode == "incorrect":
                df = df[df["correct"] == 0]
            elif mode == "all":
                pass
            else:
                raise ValueError(mode)

            dfs[d] = df

        conf_object = dfs["object"]["conf_top1"].mean()
        conf_ooc = dfs["ooc"]["conf_top1"].mean()

        gap = conf_object - conf_ooc

        results.append({
            "model": model,
            "gap": gap
        })

    return pd.DataFrame(results)


def get_group(f):
        if "blur" in f:
            return "blur"
        elif "cm" in f:
            return "cm"
        elif f == "crop":
            return "crop"
        else:
            return "base"
        
        
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