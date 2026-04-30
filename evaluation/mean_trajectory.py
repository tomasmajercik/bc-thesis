import sys
import os

# Ensure project root is on sys.path when run standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

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


def compute_mean_trajectory(train_loader) -> np.ndarray:
    """
    Compute the per-timestep mean position over all training trajectories.
    Invalid/padded steps (coord < 0) are excluded from the average.
    Returns shape (n_future, 2).
    """
    sum_coords = None
    counts = None

    for batch in train_loader:
        *_, target, coords, past_coords = batch  # coords: (B, n_future, 2)
        coords_np = coords.numpy()

        if sum_coords is None:
            n_future = coords_np.shape[1]
            sum_coords = np.zeros((n_future, 2), dtype=np.float64)
            counts = np.zeros(n_future, dtype=np.int64)

        for b in range(coords_np.shape[0]):
            for t in range(coords_np.shape[1]):
                if coords_np[b, t, 0] >= 0:  # valid (not padding)
                    sum_coords[t] += coords_np[b, t]
                    counts[t] += 1

    if sum_coords is None:
        raise RuntimeError("Training loader was empty — cannot compute mean trajectory.")

    mean_traj = np.zeros_like(sum_coords)
    last_valid = None
    for t in range(len(counts)):
        if counts[t] > 0:
            mean_traj[t] = sum_coords[t] / counts[t]
            last_valid = mean_traj[t].copy()
        elif last_valid is not None:
            # carry forward if no training samples had this timestep
            mean_traj[t] = last_valid

    return mean_traj


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


def run_mean_trajectory(
    dataset_name: str,
    split: str = "val",
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
) -> dict:
    DatasetClass = DATASETS[dataset_name]
    dataset = DatasetClass(return_coords=True, return_past_coords=True)
    train_ds, val_ds, test_ds = split_ds_sequential(dataset, train_ratio, val_ratio)
    split_map = {"train": train_ds, "val": val_ds, "test": test_ds, "eval": ConcatDataset([val_ds, test_ds])}

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
    eval_loader  = DataLoader(split_map[split], batch_size=1, shuffle=False, num_workers=0)

    print("  Computing mean trajectory from training split…")
    mean_traj = compute_mean_trajectory(train_loader)

    fde_m = FDEMetric()
    mr_m  = MRMetric(threshold_px=20.0)
    ngp_m = NearestGTPixelMetric()
    da_m  = DirectionalAccuracyMetric()
    plr_m = PathLengthRatioMetric()
    pc_m  = PathCoverageMetric()

    accs = {"FDE": [], "MR": [], "NGP": [], "DA": [], "PLR": [], "PC": []}

    with torch.no_grad():
        for batch in eval_loader:
            *_, target, coords, past_coords = batch
            B, _, H, W = target.shape

            for b in range(B):
                n_future = coords.shape[1]
                pred_hm = coords_to_heatmap(mean_traj[:n_future], H, W).unsqueeze(0)  # (1, 1, H, W)

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

    print(f"Evaluating mean trajectory baseline on {DATASET}/{SPLIT}…")
    results = run_mean_trajectory(DATASET, SPLIT)

    print(f"\n{'─' * 40}")
    for name, val in results.items():
        direction = "(↑)" if HIGHER_IS_BETTER[name] else "(↓)"
        print(f"  {name:<6} {direction} : {val:.4f}")
    print(f"{'─' * 40}")
