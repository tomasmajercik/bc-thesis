"""
kalman_baseline.py — Constant-velocity Kalman filter baseline for trajectory prediction.

Self-contained: loads datasets, runs the filter, converts predicted trajectories
to heatmaps, and evaluates all 7 metrics.

Usage:
    python kalman_baseline.py [--dataset pets] [--split test]
    python kalman_baseline.py --all_datasets          # evaluate every dataset
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from filterpy.kalman import KalmanFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.datasets import (
    PETSDataset, PETSDatasetLT,
    StMarcDataset,
    SherbrookeDataset,
    AtriumDataset,
    RouenDataset,
    MOTS16_02Dataset,
)
from training.utils import split_ds_sequential
from training.metrics import (
    EMDMetric,
    KLDMetric,
    NSSMetric,
    FDEMetric,
    MRMetric,
    NLLMetric,
    TopKCoverageMetric,
)


# ──────────────────────────────────────────────────────────────────────────────
# Kalman filter  (filterpy.kalman.KalmanFilter — Labbe, 2015)
# ──────────────────────────────────────────────────────────────────────────────

def predict_future(past_coords: np.ndarray, n_future: int,
                   process_noise: float = 1.0, obs_noise: float = 10.0) -> np.ndarray:
    """
    Constant-velocity Kalman filter using filterpy.kalman.KalmanFilter
    (Labbe, R. R., 2015. Kalman and Bayesian Filters in Python. GitHub.).

    State  : [x, y, vx, vy]
    Obs    : [x, y]

    Args:
        past_coords  : (T, 2) float array; rows where x<0 or y<0 are invalid.
        n_future     : number of future steps to extrapolate.
        process_noise: scales the process noise matrix Q.
        obs_noise    : scales the measurement noise matrix R.

    Returns:
        (n_future, 2) predicted [x, y] positions.
    """
    valid = past_coords[np.all(past_coords >= 0, axis=1)]
    if len(valid) == 0:
        return np.zeros((n_future, 2), dtype=np.float64)

    kf = KalmanFilter(dim_x=4, dim_z=2)

    # State-transition matrix (constant velocity, dt=1)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float64)

    # Observation matrix
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]], dtype=np.float64)

    kf.Q = np.eye(4) * process_noise   # process noise
    kf.R = np.eye(2) * obs_noise       # measurement noise
    kf.P = np.eye(4) * 50.0            # large initial state uncertainty

    # Initialise state from first valid observation
    vx0, vy0 = (valid[1] - valid[0]) if len(valid) >= 2 else (0.0, 0.0)
    kf.x = np.array([[valid[0, 0]], [valid[0, 1]], [vx0], [vy0]], dtype=np.float64)

    # Filter pass over all observations
    for obs in valid:
        kf.predict()
        kf.update(obs.reshape(2, 1))

    # Extrapolate into the future (no more updates)
    future = np.empty((n_future, 2), dtype=np.float64)
    for i in range(n_future):
        kf.predict()
        future[i] = kf.x[:2, 0]

    return future


# ──────────────────────────────────────────────────────────────────────────────
# Heatmap generation
# ──────────────────────────────────────────────────────────────────────────────

def coords_to_heatmap(
    coords: np.ndarray,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Splat trajectory coordinates onto a heatmap with Gaussian blobs,
    using the same growing-sigma schedule as the ground-truth generation

    This makes uncertainty grow linearly with prediction horizon, matching
    the training targets exactly so the comparison is fair.

    Args:
        coords : (T, 2) float array of predicted [x, y] positions.
        H, W   : heatmap height and width.

    Returns:
        (1, H, W) float tensor in [0, 1].
    """
    heatmap = np.zeros((H, W), dtype=np.float32)
    T = len(coords)

    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)  # both (H, W)

    for t, (x, y) in enumerate(coords):
        x = float(np.clip(x, 0, W - 1))
        y = float(np.clip(y, 0, H - 1))

        sigma = 5.0 + 4.0 * (t / max(1, T - 1))
        blob = np.exp(-((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2.0 * sigma ** 2))
        heatmap += blob

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ──────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "pets":       PETSDatasetLT,
    # "stmarc":     StMarcDataset,
    # "sherbrooke": SherbrookeDataset,
    # "atrium":     AtriumDataset,
    # "rouen":      RouenDataset,
    # "mots16_02":  MOTS16_02Dataset,
}


HIGHER_IS_BETTER = {
    "EMD":  False,
    "KLD":  False,
    "NSS":  True,
    "FDE":  False,
    "MR":   False,
    "NLL":  False,
    "TOPk": True,
}


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    dataset_name: str,
    split: str,
    train_ratio: float,
    val_ratio: float,
    proc_noise: float,
    obs_noise: float,
):
    DatasetClass = DATASETS[dataset_name]

    try:
        dataset = DatasetClass(
            return_coords=True,
            return_past_coords=True,
        )
    except Exception as exc:
        print(f"  [SKIP] {dataset_name}: could not load — {exc}")
        return None

    train_ds, val_ds, test_ds = split_ds_sequential(dataset, train_ratio, val_ratio)
    split_map = {"train": train_ds, "val": val_ds, "test": test_ds}
    eval_ds = split_map[split]

    if len(eval_ds) == 0:
        print(f"  [SKIP] {dataset_name}: {split} split is empty")
        return None

    loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=0)

    emd_m  = EMDMetric()
    kld_m  = KLDMetric()
    nss_m  = NSSMetric()
    fde_m  = FDEMetric()
    mr_m   = MRMetric(threshold_px=20.0)
    nll_m  = NLLMetric()

    accumulators = {"EMD": [], "KLD": [], "NSS": [],
                    "FDE": [], "MR":  [], "NLL": [], "TOPk": []}

    with torch.no_grad():
        for batch in loader:
            # (past, impass, ctx, zoom, target, coords, past_coords)
            *_, target, coords, past_coords = batch

            B, _, H, W = target.shape

            for b in range(B):
                pc = past_coords[b].numpy()          # (no_steps, 2)
                n_future = coords.shape[1]            # no_steps for future

                future_pred = predict_future(pc, n_future, proc_noise, obs_noise)
                pred_hm = coords_to_heatmap(future_pred, H, W)
                pred_hm = pred_hm.unsqueeze(0)       # (1, 1, H, W)

                gt_target = target[b : b + 1]        # (1, 1, H, W)
                gt_coords = coords[b : b + 1]        # (1, no_steps, 2)

                accumulators["EMD"].append(emd_m(pred_hm, gt_target).item())
                accumulators["KLD"].append(kld_m(pred_hm, gt_target).item())
                accumulators["NSS"].append(nss_m(pred_hm, gt_target, gt_coords).item())
                accumulators["FDE"].append(fde_m(pred_hm, gt_target, gt_coords).item())
                accumulators["MR"].append(mr_m(pred_hm, gt_target, gt_coords).item())
                accumulators["NLL"].append(nll_m(pred_hm, gt_target, gt_coords).item())
                accumulators["TOPk"].append(topk_m(pred_hm, gt_target, gt_coords).item())

    n = len(accumulators["EMD"])
    results = {name: float(sum(vals) / len(vals)) for name, vals in accumulators.items()}
    results["n"] = n
    return results


def print_results(dataset_name: str, split: str, results: dict):
    n = results["n"]
    print(f"\n{'─' * 46}")
    print(f"  Kalman Filter Baseline  |  {dataset_name} / {split}")
    print(f"  n_samples : {n}")
    print(f"{'─' * 46}")
    print(f"  EMD           (↓) : {results['EMD']:.4f}")
    print(f"  KLD           (↓) : {results['KLD']:.4f}")
    print(f"  NSS           (↑) : {results['NSS']:.4f}")
    print(f"  FDE           (↓) : {results['FDE']:.4f}")
    print(f"  MR  @20px     (↓) : {results['MR']:.4f}")
    print(f"  NLL           (↓) : {results['NLL']:.4f}")
    print(f"  TOPk (k=0.10) (↑) : {results['TOPk']:.4f}")
    print(f"{'─' * 46}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kalman filter baseline evaluation")
    parser.add_argument(
        "--dataset", default="pets", choices=list(DATASETS.keys()),
        help="Dataset to evaluate (ignored when --all_datasets is set)",
    )
    parser.add_argument(
        "--all_datasets", action="store_true",
        help="Run evaluation on every dataset",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--proc_noise", type=float, default=1.0,
        help="Kalman filter process noise Q = I * proc_noise",
    )
    parser.add_argument(
        "--obs_noise", type=float, default=10.0,
        help="Kalman filter observation noise R = I * obs_noise",
    )
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio",   type=float, default=0.20)
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if args.all_datasets else [args.dataset]

    for ds_name in targets:
        print(f"\nEvaluating {ds_name} ...")
        results = evaluate(
            dataset_name=ds_name,
            split=args.split,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            proc_noise=args.proc_noise,
            obs_noise=args.obs_noise,
        )
        if results is not None:
            print_results(ds_name, args.split, results)

            out_dir = "previews/metrics"
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "kalman.json")
            with open(out_path, "w") as f:
                json.dump({**results, "higher_is_better": HIGHER_IS_BETTER}, f, indent=2)
            print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()