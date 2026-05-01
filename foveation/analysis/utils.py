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
        "blur-medium",
        "blur-strong",
        "cm-light",
        "cm-medium",
        "cm-strong",
]

FOVEATION_PALETTE = {
    "base": "#4D4D4D",     
    "crop": "#E69F00",     
    "blur": "#4C956C",     
    "cm": "#9D4EDD",     
}

OOC_DATASET_ORDER = ["Original", "Object-Only", "OOC"]

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
    

def get_foveation_palette():
    return {
        # base
        "base": "#4D4D4D",
        # crop 
        "crop": "#E69F00",
        # blur
        "blur-light":  "#A3C9A8",
        "blur-medium": "#4C956C",
        "blur-strong": "#1B4332",
        # cm
        "cm-light":  "#CDB4DB",
        "cm-medium": "#9D4EDD",
        "cm-strong": "#5A189A",
        # without strengths
        "blur": "#4C956C",
        "cm": "#9D4EDD",
    }
    
    
def format_label(name):
    if "-" in name:
        base, strength = name.split("-")
        return f"{base}\n{strength}"
    return name


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
    
    
def parse_hard(name):
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