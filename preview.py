import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from model.model import MultiEncoderUNet
from training.utils import load_params, split_ds_sequential
from training.datasets import (
    PETSDataset, PETSDatasetLT, PETSDatasetST, PETSDatasetLW, PETSDatasetSW,
    StMarcDataset, SherbrookeDataset, AtriumDataset, RouenDataset, MOTS16_02Dataset, PETS09NoGauss5sec
)

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


def _load_dataset(CFG):
    use_motion = CFG.get('use_motion', False)
    kwargs = dict(
        scale=CFG['image_scale'],
        return_coords=True,
        return_past_coords=use_motion,
    )
    ds_name = CFG['dataset']
    # ds_name = "rouen" # DEBUG
    if ds_name == "pets":
        return PETSDatasetLT(**kwargs) #type: ignore
        # return PETS09NoGauss5sec(**kwargs) #type: ignore
    elif ds_name == "stmarc":
        return StMarcDataset(**kwargs) #type: ignore
    elif ds_name == "sherbrooke":
        return SherbrookeDataset(**kwargs) #type: ignore
    elif ds_name == "atrium":
        return AtriumDataset(**kwargs) #type: ignore
    elif ds_name == "rouen":
        return RouenDataset(**kwargs) #type: ignore
    elif ds_name == "mots16_02":
        return MOTS16_02Dataset(scale=(CFG['image_scale'] - 0.15), return_coords=True, return_past_coords=use_motion)
    else:
        raise ValueError(f"Unknown dataset: {ds_name!r}")


def preview(
    checkpoint_name: str,
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
    dataset = _load_dataset(CFG)
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

    ckpt_path = Path("checkpoints") / checkpoint_name / "best_model.pth"
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

            from matplotlib.patches import Patch
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


if __name__ == "__main__":
    preview("TverskyLossBalanced", num_images=30)
