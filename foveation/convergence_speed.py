import torch
import torch.nn.functional as F
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

from foveation.ooc.ooc_utils import MODEL_CONFIGS
from foveation.analysis.utils import set_thesis_style, parse_hard, get_foveation_palette


# CONFIG
BASE_DIR = Path("/home/data/elias/archive_extracted/mocov3")
FIG_DIR = Path("/home/elias/solo-learn/foveation/analysis/outputs/figures/representations")
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["base", "crop", "blur-light", "blur", "blur-strong", "cm-light", "cm", "cm-strong"]


# HELPERS
def load_state(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Loaded: {ckpt_path}.")
    state = ckpt["state_dict"]
    cleaned = {}
    for k in state:
        new_k = k
        if "encoder" in k:
            new_k = k.replace("encoder", "backbone")
        if "backbone." in new_k:
            new_k = new_k.replace("backbone.", "")
        cleaned[new_k] = state[k]
    print(f"Finished loading: {ckpt_path}.")
    return cleaned

def flatten_state(state_dict):
    vec = []
    for k in sorted(state_dict.keys()):
        v = state_dict[k]
        if torch.is_tensor(v):
            vec.append(v.flatten())
    return torch.cat(vec)

def cosine_sim(a, b):
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return torch.dot(a, b).item()

# these l2 norms yielded overlapping straight lines as a result
def l2_dist(a, b):
    return torch.norm(a - b).item()

def l2_dist_normalized(a, b):
    return torch.norm(a - b) / torch.norm(b)

def get_ckpts(model_cfg):
    run_id = model_cfg["id"]
    name = model_cfg["name"]
    run_dir = BASE_DIR / run_id
    ckpts = list(run_dir.glob("*.ckpt"))
    # extract epoch
    def get_epoch(p):
        if "last" in p.name:
            return 30
        m = re.search(r"ep=(\d+)", p.name)
        return int(m.group(1)) if m else -1
    ckpts = sorted(ckpts, key=get_epoch)
    return ckpts

# MAIN ANALYSIS
def analyze_model(model_cfg):
    ckpts = get_ckpts(model_cfg)
    # final = last checkpoint
    final_ckpt = [c for c in ckpts if "last" in c.name][0]
    final_state = flatten_state(load_state(final_ckpt))
    epochs = []
    sims = []
    for ckpt in ckpts:
        if "last" in ckpt.name:
            continue
        epoch = int(re.search(r"ep=(\d+)", ckpt.name).group(1))
        state = flatten_state(load_state(ckpt))
        sim = cosine_sim(state, final_state)
        epochs.append(epoch)
        sims.append(sim)
    return epochs, sims


def plot():

    set_thesis_style()
    
    plt.figure(figsize=(7, 4))
    palette = get_foveation_palette()
    
    for m in MODEL_CONFIGS:
        name = parse_hard(m["name"])
        # filter models you want
        if not any(x in name for x in MODELS):
            continue
        epochs, sims = analyze_model(m)
        print(f"Finished analysing {name}.")
        if name == "base":
            plt.plot(epochs, sims, marker="o", label=name, color=palette.get(name, "black"), linewidth=2, zorder=10)
        else:
            plt.plot(
                epochs, sims, marker="o", label=name, color=palette.get(name, "black"), linewidth=1.5, alpha=0.8
            )
    plt.xlabel("Epoch")
    plt.ylabel("Cosine Similarity to Final Weights")
    plt.title("Convergence Speed of Backbones")
    plt.legend()
    plt.tight_layout()
    
    out_path = FIG_DIR / f"convergence speed.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved → {out_path}")
    

if __name__ == "__main__":
    plot()


