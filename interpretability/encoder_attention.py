import os
import torch
import matplotlib.pyplot as plt
import pandas as pd

from model.model import MultiEncoderUNet
from training.datasets import PETSDataset
from torch.utils.data import DataLoader, Subset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("previews/interpretability/attention", exist_ok=True)

# ===============================
# Dataset
# ===============================
sample_indices = [1303]
dataset = PETSDataset(scale=0.5)
subset = Subset(dataset, sample_indices)
samples = DataLoader(subset, batch_size=1, shuffle=False)

# ===============================
# Load model
# ===============================
model = MultiEncoderUNet().to(DEVICE)
ckpt_path = "checkpoints/long-strict-w-imgs/best_model.pth"
checkpoint = torch.load(ckpt_path, map_location=DEVICE)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

modalities_per_level = [
    ["Past", "Obstacle", "Context", "Zoom"],
    ["Past", "Obstacle", "Context", "Zoom"],
    ["Past", "Obstacle", "Context", "Zoom"],
    ["Context", "Zoom"]
]

# ===============================
# Run sample
# ===============================
for idx, batch in enumerate(samples):

    with torch.no_grad():
        past, imp, ctx, zoom, target = [
            x.to(DEVICE, non_blocking=True) for x in batch
        ]

        out, att_weights = model(
            past, imp, ctx, zoom,
            return_attention=True
        )

    print("\n==============================")
    print(f"Sample index: {sample_indices[idx]}")
    print("==============================\n")

    # ===============================
    # Create single figure (2x2)
    # ===============================
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for lvl, weights in enumerate(att_weights):

        w = weights[0].detach().cpu()

        # ---- PRINT TABLE ----
        df = pd.DataFrame({
            "Modality": modalities_per_level[lvl],
            "Weight": w.numpy()
        })

        print(f"\nLevel {lvl+1} Attention Weights")
        print(df.to_string(index=False))

        # ---- SUBPLOT ----
        ax = axes[lvl]
        ax.bar(modalities_per_level[lvl], w)
        ax.set_ylim(0, 1)
        ax.set_title(f"Level {lvl+1}")
        ax.set_ylabel("Weight")

    fig.suptitle("Encoder Attention Weights per Level", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # type: ignore

    save_path = f"previews/interpretability/attention/attention_all_levels_sample_{sample_indices[idx]}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\nSaved combined plot → {save_path}")

## run with python -m interpretability.encoder_attention