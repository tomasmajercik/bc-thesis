import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_activation_grid(activations: dict[str, torch.Tensor], ios: tuple, mean_across_channels: bool = True, save_path="grid.png"):
    """
    activations: dict of name -> tensor [B, C, H, W]
    ios: tuple of (past, imp, ctx, zoom, target, out), each [B, C, H, W]
    """
    past, imp, ctx, zoom, target, out = ios

    # Define rows: (row_label, io_tensor, [activation_key, ...])
    rows = [
        ("Past",       past,  sorted([k for k in activations if k.startswith("past")])),
        ("Obstacle", imp,   sorted([k for k in activations if k.startswith("obstacle")])),
        ("Context",    ctx,   sorted([k for k in activations if k.startswith("context")])),
        ("Zoom",       zoom,  sorted([k for k in activations if k.startswith("zoom")])),
        ("Target/Out", None,  []),  # special row
    ]

    n_cols = max(1 + len(r[2]) for r in rows[:4])  # io + activations
    n_cols = max(n_cols, 2)                          # target/out row needs at least 2
    n_rows = len(rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    def render_input(ax, tensor, title):
        """Render a raw input tensor (gray or RGB)."""
        img_tensor = tensor[0]  # first batch item
        if img_tensor.shape[0] > 1:
            img = img_tensor.permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
        else:
            img = img_tensor[0].cpu().numpy()
            ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    def render_heatmap(ax, tensor, title):
        """Render a heatmap output (target or prediction) matching wandb style."""
        img = tensor[0, 0].cpu().numpy() * 255.0
        ax.imshow(img, cmap="hot", vmin=0, vmax=255)
        ax.set_title(f"{title} (max={img.max():.2f}, mean={img.mean():.4f})", fontsize=8)
        ax.axis("off")

    def render_feat(ax, feat, title):
        """Render mean activation map."""
        if mean_across_channels:
            img = feat[0].mean(dim=0).cpu().numpy()
        else:
            img = feat[0].max(dim=0)[0].cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img, cmap="viridis")
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    def blank(ax):
        ax.axis("off")

    # Rows 0-3: input image -> encoder activations
    for row_idx, (label, io_tensor, act_keys) in enumerate(rows[:4]):
        ax_row = axes[row_idx]
        render_input(ax_row[0], io_tensor, label)
        for col_idx, key in enumerate(act_keys, start=1):
            render_feat(ax_row[col_idx], activations[key], key)
        for col_idx in range(1 + len(act_keys), n_cols):
            blank(ax_row[col_idx])

    # Last row: target and prediction side by side, rest blank
    ax_row = axes[4]
    render_heatmap(ax_row[0], target, "Target")
    render_heatmap(ax_row[1], out, "Prediction")
    for col_idx in range(2, n_cols):
        blank(ax_row[col_idx])

    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\033[32m[INFO]\033[0m Saved activation grid to {save_path}")
    plt.close()