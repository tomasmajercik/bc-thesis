import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from torch.utils.data import DataLoader

from model.model import MultiEncoderUNet
from training.utils import load_params, split_ds_sequential
from training.datasets import PetsDataset, RouenDataset, StMarcDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _overlay(ax, heatmap, color_rgb, label, alpha_scale=1.0):
    """Overlay a heatmap on an existing axes using a solid colour with heatmap-driven alpha."""
    h = np.clip(heatmap, 0.0, 1.0)
    rgba = np.zeros((*h.shape, 4), dtype=np.float32)
    rgba[..., 0] = color_rgb[0]
    rgba[..., 1] = color_rgb[1]
    rgba[..., 2] = color_rgb[2]
    rgba[..., 3] = h * alpha_scale
    ax.imshow(rgba, interpolation='bilinear')


def _load_dataset(CFG, dataset_name, force_past_coords=False):
    use_motion = CFG.get('use_motion', False)
    kwargs = dict(
        scale=CFG['image_scale'],
        return_coords=True,
        return_past_coords=use_motion or force_past_coords,
    )
    ds_map = {
        "pets":   PetsDataset,
        "rouen":  RouenDataset,
        "stmarc": StMarcDataset,
    }
    if dataset_name not in ds_map:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")
    return ds_map[dataset_name](**kwargs)  # type: ignore


def preview(
    checkpoint_name: str,
    epoch: int,
    dataset_name: str,
    num_images: int = 30,
    config_path: str = "training/config/training_cfg.yaml",
):
    """
    Load a checkpoint, run inference on the val split, and save overlay plots to
    previews/model_io/<checkpoint_name>/id{0..num_images-1}.png

    Each image shows three heatmaps overlaid on the context frame:
      - Blue  : past trajectory raster
      - Green : ground-truth future density
      - Red   : model prediction
    """
    CFG      = load_params(config_path)
    use_motion = CFG.get('use_motion', False)

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = _load_dataset(CFG, dataset_name)
    _, val_ds, _ = split_ds_sequential(dataset, CFG['train_ratio'], CFG['val_ratio'])
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MultiEncoderUNet(
        past_channels     = 1,
        obstacle_channels = 1,
        context_channels  = 3,
        zoom_channels     = 3,
        width             = CFG['model_size'],
        use_motion          = use_motion,
    ).to(DEVICE)

    if epoch is None:
        ckpt_path = Path("checkpoints") / checkpoint_name / "best_model.pth"
    else:
        ckpt_path = Path("checkpoints") / checkpoint_name / f"[{epoch}]_epoch.pth"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # ── Output dir ───────────────────────────────────────────────────────────
    out_dir = Path("previews/model_io") / checkpoint_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Inference & plotting ──────────────────────────────────────────────────
    saved = 0
    with torch.no_grad():
        for batch in val_loader:
            if saved >= num_images:
                break

            if use_motion:
                past, imp, ctx, zoom, target, coords, past_coords = [x.to(DEVICE) for x in batch]
                pred = model(past, imp, ctx, torch.zeros_like(zoom), past_coords)
            else:
                past, imp, ctx, zoom, target, coords = [x.to(DEVICE) for x in batch]
                pred = model(past, imp, ctx, torch.zeros_like(zoom))

            ctx_np  = ctx[0].permute(1, 2, 0).cpu().numpy()          # (H, W, 3)
            past_np = past[0, 0].cpu().numpy()                        # (H, W)
            gt_np   = target[0, 0].cpu().numpy()                      # (H, W)
            pred_np = pred[0, 0].cpu().numpy()                        # (H, W) — model has Sigmoid inside OutConv

            fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
            ax.imshow(ctx_np)
            _overlay(ax, past_np, (0.2, 0.6, 1.0), "past",  alpha_scale=0.8)   # blue
            _overlay(ax, gt_np,   (0.0, 1.0, 0.2), "GT",    alpha_scale=0.85)  # green
            _overlay(ax, pred_np, (1.0, 0.15, 0.1), "pred", alpha_scale=0.85)  # red

            legend = [
                Patch(color=(0.2, 0.6, 1.0), label="Past trajectory"),
                Patch(color=(0.0, 1.0, 0.2), label="Ground truth"),
                Patch(color=(1.0, 0.15, 0.1), label="Prediction"),
            ]
            ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.6)
            ax.axis('off')
            ax.set_title(f"{checkpoint_name}  —  sample {saved}", fontsize=9)

            fig.savefig(out_dir / f"id{saved}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved += 1

    print(f"Saved {saved} previews → {out_dir}")


def preview_with_kalman(
    checkpoint_name: str,
    epoch: int,
    dataset_name: str,
    start: int = 0,
    end: int = 50,
    config_path: str = "training/config/training_cfg.yaml",
):
    """
    Compare model predictions against the Kalman filter baseline over a specific
    range of val-set samples [start, end).

    Each saved image is a 2-row × 1-col figure:
      Row 0 (top)    : model prediction
      Row 1 (bottom) : Kalman filter prediction
    Both rows share the same overlay scheme:
      Blue  — past trajectory raster
      Green — ground-truth future density
      Red   — prediction (model or Kalman, respectively)

    Output: previews/model_io/model-kalman-comparison/<checkpoint_name>/id{i}.png
    Files are named by their actual sample index so ranges are comparable across runs.
    """
    from evaluation.kalman_baseline import predict_future, coords_to_heatmap

    CFG = load_params(config_path)
    use_motion = CFG.get('use_motion', False)

    # Always need past_coords for the Kalman filter regardless of use_motion
    dataset = _load_dataset(CFG, dataset_name, force_past_coords=True)
    _, val_ds, _ = split_ds_sequential(dataset, CFG['train_ratio'], CFG['val_ratio'])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = MultiEncoderUNet(
        past_channels=1,
        obstacle_channels=1,
        context_channels=3,
        zoom_channels=3,
        width=CFG['model_size'],
        use_motion=use_motion,
    ).to(DEVICE)

    if epoch is None:
        ckpt_path = Path("checkpoints") / checkpoint_name / "best_model.pth"
    else:
        ckpt_path = Path("checkpoints") / checkpoint_name / f"[{epoch}]_epoch.pth"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    out_dir = Path("previews/model_io/model-kalman-comparison") / checkpoint_name
    out_dir.mkdir(parents=True, exist_ok=True)

    legend_patches = [
        Patch(color=(0.2, 0.6, 1.0), label="Past trajectory"),
        Patch(color=(0.0, 1.0, 0.2), label="Ground truth"),
        Patch(color=(1.0, 0.15, 0.1), label="Prediction"),
    ]

    saved = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if idx < start:
                continue
            if idx >= end:
                break

            # past_coords is always present because force_past_coords=True
            if use_motion:
                past, imp, ctx, zoom, target, coords, past_coords = [x.to(DEVICE) for x in batch]
                pred = model(past, imp, ctx, torch.zeros_like(zoom), past_coords)
            else:
                past, imp, ctx, zoom, target, coords, past_coords = [x.to(DEVICE) for x in batch]
                pred = model(past, imp, ctx, torch.zeros_like(zoom))

            ctx_np  = ctx[0].permute(1, 2, 0).cpu().numpy()   # (H, W, 3)
            past_np = past[0, 0].cpu().numpy()                 # (H, W)
            gt_np   = target[0, 0].cpu().numpy()               # (H, W)
            pred_np = pred[0, 0].cpu().numpy()                 # (H, W)
            H, W    = gt_np.shape

            # Kalman prediction
            pc_np       = past_coords[0].cpu().numpy()
            n_future    = coords.shape[1]
            future_pred = predict_future(pc_np, n_future)
            kalman_np   = coords_to_heatmap(future_pred, H, W)[0].numpy()  # (H, W)

            fig, axes = plt.subplots(
                2, 1,
                figsize=(6, 10),
                tight_layout=True,
                gridspec_kw={"hspace": 0.08},
            )

            # ── Top: model ────────────────────────────────────────────────────
            ax = axes[0]
            ax.imshow(ctx_np)
            _overlay(ax, past_np, (0.2, 0.6, 1.0), "past",  alpha_scale=0.8)
            _overlay(ax, gt_np,   (0.0, 1.0, 0.2), "GT",    alpha_scale=0.85)
            _overlay(ax, pred_np, (1.0, 0.15, 0.1), "pred", alpha_scale=0.85)
            ax.legend(handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.6)
            ax.axis('off')
            ax.set_title(f"Model - sample {idx}", fontsize=8)

            # ── Bottom: Kalman ────────────────────────────────────────────────
            ax = axes[1]
            ax.imshow(ctx_np)
            _overlay(ax, past_np,  (0.2, 0.6, 1.0), "past",   alpha_scale=0.8)
            _overlay(ax, gt_np,    (0.0, 1.0, 0.2), "GT",     alpha_scale=0.85)
            _overlay(ax, kalman_np,(1.0, 0.15, 0.1), "kalman", alpha_scale=0.85)
            ax.legend(handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.6)
            ax.axis('off')
            ax.set_title(f"Kalman - sample {idx}", fontsize=8)

            fig.savefig(out_dir / f"id{idx}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved += 1

    print(f"Saved {saved} previews → {out_dir}")


if __name__ == "__main__":
    # preview("pets-balanced2", epoch=16, dataset_name="pets", num_images=30) # 5-7-*8*-16
    # preview("rouen-balanced2", epoch=19, dataset_name="rouen", num_images=30) # 8-*19*
    # preview("stmarc-balanced", epoch=12, dataset_name="stmarc", num_images=30) # *12*

    preview_with_kalman("pets-balanced2", epoch=9, dataset_name="pets", start=300, end=400)
    # preview_with_kalman("stmarc-balanced", epoch=12, dataset_name="stmarc", start=800, end=1000)
    # preview_with_kalman("rouen-balanced2", epoch=19, dataset_name="rouen", start=400, end=500)
