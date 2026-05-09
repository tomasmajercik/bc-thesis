import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model.model import MultiEncoderUNet
from training.datasets import PetsDataset, StMarcDataset, RouenDataset
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 500
# CKPT_PATH = "checkpoints/pets-balanced2/[15]_epoch.pth"
CKPT_PATH = "checkpoints/rouen-balanced2/[9]_epoch.pth"
OUT_DIR   = "previews/interpretability/attention"
os.makedirs(OUT_DIR, exist_ok=True)

MODALITIES_PER_LEVEL = [
    ["Past", "Obstacle", "Context", "Zoom"],
    ["Past", "Obstacle", "Context", "Zoom"],
    ["Past", "Obstacle", "Context", "Zoom"],
    ["Context", "Zoom"],
]

# ── Dataset ───────────────────────────────────────────────────────────────────
# dataset = PetsDataset(scale=0.5)
# dataset = StMarcDataset(scale=0.5)
dataset = RouenDataset(scale=0.5)
loader  = DataLoader(dataset, batch_size=1, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
model = MultiEncoderUNet().to(DEVICE)
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ── Accumulate attention weights over N_SAMPLES ───────────────────────────────
n_levels = len(MODALITIES_PER_LEVEL)
acc = [np.zeros(len(MODALITIES_PER_LEVEL[lvl]), dtype=np.float64) for lvl in range(n_levels)]
count = 0

with torch.no_grad():
    for batch in loader:
        if count >= N_SAMPLES:
            break
        past, imp, ctx, zoom, target = [x.to(DEVICE, non_blocking=True) for x in batch]
        _, att_weights = model(past, imp, ctx, zoom, return_attention=True)
        for lvl, weights in enumerate(att_weights):
            acc[lvl] += weights[0].detach().cpu().numpy()
        count += 1

avg = [acc[lvl] / count for lvl in range(n_levels)]

print(f"\nAveraged attention weights over {count} samples:")

# Build a table — pad missing modalities with NaN so all rows are the same width
all_modalities = sorted({m for lvl in MODALITIES_PER_LEVEL for m in lvl})
rows = {}
for lvl, w in enumerate(avg):
    rows[f"Level {lvl + 1}"] = {m: v for m, v in zip(MODALITIES_PER_LEVEL[lvl], w)}
df = pd.DataFrame(rows, index=all_modalities).T
df.index.name = "Level"
print(df.to_string(float_format=lambda x: f"{x:.4f}"))


# ── Plot helper ───────────────────────────────────────────────────────────────
def _save_figure(weights_per_level, path, log_scale=False):
    scale_tag = "log-normalised" if log_scale else "linear"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for lvl, w in enumerate(weights_per_level):
        if log_scale:
            w = np.log1p(w)
            m = w.max()
            w = w / m if m > 0 else w

        ax = axes[lvl]
        ax.bar(MODALITIES_PER_LEVEL[lvl], w, color="steelblue")
        ax.set_ylim(0, 1)
        ax.set_title(f"Level {lvl + 1}", fontsize=11)
        ax.set_ylabel("Weight" + (" (log-norm)" if log_scale else ""))

    fig.suptitle(
        f"Encoder Attention Weights — avg over {count} samples ({scale_tag})",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # type: ignore
    fig.savefig(path, dpi=150)
    print(f"Saved → {path}")
    plt.show()
    plt.close(fig)


# ── Save both scales ──────────────────────────────────────────────────────────
_save_figure(avg, f"{OUT_DIR}/attention_avg{count}_linear.png", log_scale=False)
_save_figure(avg, f"{OUT_DIR}/attention_avg{count}_log.png",    log_scale=True)

## run with: python -m interpretability.encoder_attention
