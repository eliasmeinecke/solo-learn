from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from foveation.ooc.ooc_utils import load_model


FIG_DIR = Path("/home/elias/solo-learn/foveation/analysis/outputs/figures/representations")
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS = ["base", "crop", "cm-strong"]


def get_linear_weights(model):
    if hasattr(model, "head"):
        W = model.head.weight.data
    else:
        raise ValueError("Could not find linear head weights.")
    return W


def get_similarity(model):

    W = get_linear_weights(model)

    # normalize rows
    W = F.normalize(W, dim=1)

    # cosine similarity matrix
    sim = W @ W.T

    # remove diagonal (self-similarity)
    mask = ~torch.eye(
        sim.size(0),
        dtype=torch.bool,
        device=sim.device
    )

    sims = sim[mask]

    return sims.cpu().numpy()



sims = {}

for m in MODELS:

    print(f"Loading {m} ...")
    model = load_model(m)
    sims[m] = get_similarity(model)
    print(f"{m}: mean={np.mean(sims[m]):.4f}, std={np.std(sims[m]):.4f}")


plt.figure(figsize=(7,5))

for name, s in sims.items():

    plt.hist(
        s,
        bins=100,
        density=True,
        alpha=0.4,
        label=name
    )

plt.xlabel("Cosine Similarity Between Class Weights")
plt.ylabel("Density")
plt.title("Linear Head Class Similarity")
plt.legend()
plt.tight_layout()

out_path = FIG_DIR / f"linear_head_similarity_histogram.pdf"
plt.savefig(out_path, bbox_inches="tight")
plt.close()

print(f"Saved → {out_path}")

"""
base: mean=-0.0009, std=0.0670
crop: mean=-0.0009, std=0.0624
cm-strong: mean=-0.0009, std=0.0648
"""

