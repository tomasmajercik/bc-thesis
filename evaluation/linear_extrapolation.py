"""
linear_extrapolation.py — Linear extrapolation baseline.
Exports run_linear() for use by evaluate.py in the project root.

Not used in the final thesis, but left here for future comparison.
"""

import sys
import os

# Ensure project root is on sys.path when run standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from training.datasets import (
    PetsDataset,
    RouenDataset,
    AtriumDataset,
    SherbrookeDataset,
    StMarcDataset,
    MotDataset,
)
from training.utils import split_ds_sequential
from evaluation.metrics import (
    FDEMetric,
    MRMetric,
    NearestGTPixelMetric,
    DirectionalAccuracyMetric,
    PathLengthRatioMetric,
    PathCoverageMetric,
)

DATASETS = {
    "pets":       PetsDataset,
    "rouen":      RouenDataset,
    "atrium":     AtriumDataset,
    "sherbrooke": SherbrookeDataset,
    "stmarc":     StMarcDataset,
    "mot":        MotDataset,
}

HIGHER_IS_BETTER = {
    "FDE": False,
    "MR":  False,
    "NGP": False,
    "DA":  True,
    "PLR": True,
    "PC":  True,
}


def predict_linear(past_coords: np.ndarray, n_future: int) -> np.ndarray:
    """
    Naive linear extrapolation baseline.
    Estimates velocity from last two observed positions and extrapolates.
    No filtering, no uncertainty — dumbest possible motion model.
    """
    valid = past_coords[np.all(past_coords >= 0, axis=1)]
    if len(valid) < 2:
        return np.tile(valid[-1], (n_future, 1)) if len(valid) > 0 else np.zeros((n_future, 2))

    velocity = valid[-1] - valid[-2]
    future = np.array([valid[-1] + velocity * (t + 1) for t in range(n_future)])
    return future


def coords_to_heatmap(coords: np.ndarray, H: int, W: int) -> torch.Tensor:
    """
    Splat trajectory coordinates onto a heatmap with Gaussian blobs,
    using the same growing-sigma schedule as the ground-truth generation.
    Returns (1, H, W) float tensor in [0, 1].
    """
    heatmap = np.zeros((H, W), dtype=np.float32)
    T = len(coords)
    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    for t, (x, y) in enumerate(coords):
        x = float(np.clip(x, 0, W - 1))
        y = float(np.clip(y, 0, H - 1))
        sigma = 5.0 + 4.0 * (t / max(1, T - 1))
        blob = np.exp(-((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2.0 * sigma ** 2))
        heatmap += blob

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)


def run_linear(
    dataset_name: str,
    split: str = "val",
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
) -> dict:
    DatasetClass = DATASETS[dataset_name]
    dataset = DatasetClass(return_coords=True, return_past_coords=True)
    train_ds, val_ds, test_ds = split_ds_sequential(dataset, train_ratio, val_ratio)
    split_map = {"train": train_ds, "val": val_ds, "test": test_ds, "eval": ConcatDataset([val_ds, test_ds])}
    loader = DataLoader(split_map[split], batch_size=1, shuffle=False, num_workers=0)

    fde_m = FDEMetric()
    mr_m  = MRMetric(threshold_px=20.0)
    ngp_m = NearestGTPixelMetric()
    da_m  = DirectionalAccuracyMetric()
    plr_m = PathLengthRatioMetric()
    pc_m  = PathCoverageMetric()

    accs = {"FDE": [], "MR": [], "NGP": [], "DA": [], "PLR": [], "PC": []}

    with torch.no_grad():
        for batch in loader:
            *_, target, coords, past_coords = batch
            B, _, H, W = target.shape

            for b in range(B):
                pc_np    = past_coords[b].numpy()
                n_future = coords.shape[1]

                future_pred = predict_linear(pc_np, n_future)
                pred_hm = coords_to_heatmap(future_pred, H, W).unsqueeze(0)  # (1, 1, H, W)

                gt_target = target[b:b+1]
                gt_coords = coords[b:b+1]
                pc_tensor = past_coords[b:b+1]

                accs["FDE"].append(fde_m.forward(pred_hm, gt_target, gt_coords).item())
                accs["MR"].append(mr_m.forward(pred_hm, gt_target, gt_coords).item())
                accs["NGP"].append(ngp_m.forward(pred_hm, gt_target, gt_coords).item())
                accs["DA"].append(da_m.forward(pred_hm, gt_target, gt_coords, pc_tensor).item())
                accs["PLR"].append(plr_m.forward(pred_hm, gt_target, gt_coords, pc_tensor).item())
                accs["PC"].append(pc_m.forward(pred_hm, gt_target, gt_coords).item())

    return {name: float(sum(v) / len(v)) for name, v in accs.items()}


if __name__ == "__main__":
    DATASET = "pets"
    SPLIT   = "eval"  # "eval" = val+test combined; "val"/"test"/"train" also valid

    print(f"Evaluating linear extrapolation baseline on {DATASET}/{SPLIT}…")
    results = run_linear(DATASET, SPLIT)

    print(f"\n{'─' * 40}")
    for name, val in results.items():
        direction = "(↑)" if HIGHER_IS_BETTER[name] else "(↓)"
        print(f"  {name:<6} {direction} : {val:.4f}")
    print(f"{'─' * 40}")
