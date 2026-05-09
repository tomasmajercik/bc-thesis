"""
kalman_baseline.py — Constant-velocity Kalman filter baseline.

Exports run_kalman() for use by evaluate.py in the project root.
"""

import sys
import os

# Ensure project root is on sys.path when run standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from filterpy.kalman import KalmanFilter

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


def predict_future(past_coords: np.ndarray, n_future: int,
                   process_noise: float = 1.0, obs_noise: float = 10.0) -> np.ndarray:
    """
    Constant-velocity Kalman filter using filterpy.kalman.KalmanFilter
    (Labbe, R. R., 2015. Kalman and Bayesian Filters in Python. GitHub.).

    State  : [x, y, vx, vy]
    Obs    : [x, y]
    """
    valid = past_coords[np.all(past_coords >= 0, axis=1)]
    if len(valid) == 0:
        return np.zeros((n_future, 2), dtype=np.float64)

    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float64)
    kf.H = np.array([[1, 0, 0, 0], #type: ignore
                     [0, 1, 0, 0]], dtype=np.float64)
    kf.Q = np.eye(4) * process_noise
    kf.R = np.eye(2) * obs_noise
    kf.P = np.eye(4) * 50.0

    vx0, vy0 = (valid[1] - valid[0]) if len(valid) >= 2 else (0.0, 0.0)
    kf.x = np.array([[valid[0, 0]], [valid[0, 1]], [vx0], [vy0]], dtype=np.float64)

    for obs in valid:
        kf.predict()
        kf.update(obs.reshape(2, 1))

    future = np.empty((n_future, 2), dtype=np.float64)
    for i in range(n_future):
        kf.predict()
        future[i] = kf.x[:2, 0]

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
        sigma = 7.0 - 4.0 * (t / max(1, T - 1)) 
        blob = np.exp(-((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2.0 * sigma ** 2))
        heatmap += blob

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)


def run_kalman(
    dataset_name: str,
    split: str = "val",
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    proc_noise: float = 1.0,
    obs_noise: float = 10.0,
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
                pc_np   = past_coords[b].numpy()
#                 n_future = max(1, int(coords.shape[1] * 0.47)) # <- for shorter kalman
                n_future = coords.shape[1]                       # <- for full kalman

                future_pred = predict_future(pc_np, n_future, proc_noise, obs_noise)
                pred_hm = coords_to_heatmap(future_pred, H, W).unsqueeze(0)  # (1, 1, H, W)

                gt_target  = target[b:b+1]        # (1, 1, H, W)
                gt_coords  = coords[b:b+1]         # (1, steps, 2)
                pc_tensor  = past_coords[b:b+1]    # (1, past_steps, 2)

                accs["FDE"].append(fde_m.forward(pred_hm, gt_target, gt_coords).item())
                accs["MR"].append(mr_m.forward(pred_hm, gt_target, gt_coords).item())
                accs["NGP"].append(ngp_m.forward(pred_hm, gt_target, gt_coords).item())
                accs["DA"].append(da_m.forward(pred_hm, gt_target, gt_coords, pc_tensor).item())
                accs["PLR"].append(plr_m.forward(pred_hm, gt_target, gt_coords, pc_tensor).item())
                accs["PC"].append(pc_m.forward(pred_hm, gt_target, gt_coords).item())

    return {name: float(sum(v) / len(v)) for name, v in accs.items()}


if __name__ == "__main__":
    DATASET     = "pets"
    SPLIT       = "eval"  # "eval" = val+test combined; "val"/"test"/"train" also valid
    PROC_NOISE  = 1.0
    OBS_NOISE   = 10.0

    print(f"Evaluating Kalman baseline on {DATASET}/{SPLIT}…")
    results = run_kalman(DATASET, SPLIT, proc_noise=PROC_NOISE, obs_noise=OBS_NOISE)

    print(f"\n{'─' * 40}")
    for name, val in results.items():
        direction = "(↑)" if HIGHER_IS_BETTER[name] else "(↓)"
        print(f"  {name:<6} {direction} : {val:.4f}")
    print(f"{'─' * 40}")
