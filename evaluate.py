"""This evaluation script was written using Antropic's coding model"""
import os
import re
import json
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from model.model import MultiEncoderUNet
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
from evaluation.kalman_baseline import run_kalman
from evaluation.linear_extrapolation import run_linear
from evaluation.mean_trajectory import run_mean_trajectory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIGHER_IS_BETTER = {
    "FDE": False,
    "MR":  False,
    "NGP": False,
    "DA":  True,
    "PLR": True,
    "PC":  True,
}

DATASETS = {
    "pets":       PetsDataset,
    "rouen":      RouenDataset,
    "atrium":     AtriumDataset,
    "sherbrooke": SherbrookeDataset,
    "stmarc":     StMarcDataset,
    "mot":        MotDataset,
}


def _eval_checkpoint(ckpt_path: str, loader) -> dict:
    model = MultiEncoderUNet().to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fde_m = FDEMetric()
    mr_m  = MRMetric(threshold_px=20.0)
    ngp_m = NearestGTPixelMetric()
    da_m  = DirectionalAccuracyMetric()
    plr_m = PathLengthRatioMetric()
    pc_m  = PathCoverageMetric()

    accs = {"FDE": [], "MR": [], "NGP": [], "DA": [], "PLR": [], "PC": []}

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {os.path.basename(ckpt_path)}", leave=False):
            past, obstacle, context, zoom, target, coords, past_coords = [
                x.to(DEVICE) for x in batch
            ]
            pred = model(past, obstacle, context, zoom)

            accs["FDE"].append(fde_m.forward(pred, target, coords).item())
            accs["MR"].append(mr_m.forward(pred, target, coords).item())
            accs["NGP"].append(ngp_m.forward(pred, target, coords).item())
            accs["DA"].append(da_m.forward(pred, target, coords, past_coords).item())
            accs["PLR"].append(plr_m.forward(pred, target, coords, past_coords).item())
            accs["PC"].append(pc_m.forward(pred, target, coords).item())

    return {name: float(sum(v) / len(v)) for name, v in accs.items()}


def _select_best_epoch(all_results: dict) -> int:
    """Rank each epoch per metric (rank 0 = best); epoch with lowest total rank wins."""
    epochs = list(all_results.keys())
    rank_sum = defaultdict(float)
    for metric, hib in HIGHER_IS_BETTER.items():
        sorted_epochs = sorted(epochs, key=lambda e: all_results[e][metric], reverse=hib)
        for rank, ep in enumerate(sorted_epochs):
            rank_sum[ep] += rank
    return min(rank_sum.keys(), key=lambda e: rank_sum[e])


def run_model(
    dataset_name: str,
    checkpoint_run: str,
    split: str = "val",
    epoch: int | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    batch_size: int = 8,
) -> dict:
    DatasetClass = DATASETS[dataset_name]
    dataset = DatasetClass(return_coords=True, return_past_coords=True)
    train_ds, val_ds, test_ds = split_ds_sequential(dataset, train_ratio, val_ratio)
    split_map = {"train": train_ds, "val": val_ds, "test": test_ds, "eval": ConcatDataset([val_ds, test_ds])}
    loader = DataLoader(split_map[split], batch_size=batch_size, shuffle=False)

    run_dir = os.path.join("checkpoints", checkpoint_run)

    if epoch is not None:
        ckpt_path = os.path.join(run_dir, f"[{epoch}]_epoch.pth")
        results = _eval_checkpoint(ckpt_path, loader)
        return {"epoch": epoch, **results}

    epoch_pattern = re.compile(r'\[(\d+)\]_epoch\.pth$')
    epoch_files = [
        (int(m.group(1)), os.path.join(run_dir, fname))
        for fname in os.listdir(run_dir)
        if (m := epoch_pattern.search(fname))
    ]

    if not epoch_files:
        for fallback in ("best_model.pth", "best_model.pt"):
            p = os.path.join(run_dir, fallback)
            if os.path.exists(p):
                print(f"  No epoch checkpoints found — using {fallback}")
                results = _eval_checkpoint(p, loader)
                return {"epoch": "best_model", **results}
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    print(f"  Found {len(epoch_files)} epoch checkpoints, evaluating all…")
    all_results = {}
    for ep_num, f in sorted(epoch_files):
        all_results[ep_num] = _eval_checkpoint(f, loader)

    best = _select_best_epoch(all_results)
    print(f"  Best epoch: {best}")
    return {"epoch": best, **all_results[best]}


if __name__ == "__main__":
    start_time = time.time()
    DATASET        = "pets"
    # DATASET        = "rouen"
    # DATASET        = "stmarc"
    CHECKPOINT_RUN = "pets-balanced2"
    # CHECKPOINT_RUN = "rouen-balanced2"
    # CHECKPOINT_RUN = "stmarc-balanced"

    EPOCH          = None   # int → run only that epoch; None → run all, keep best
    SPLIT          = "eval"  # "eval" = val+test combined; "val"/"test"/"train" also valid
    BATCH_SIZE     = 4

    print("*" * 10)
    print(f"{DATASET}:{CHECKPOINT_RUN}")
    print("*" * 10)
    print("Running Kalman baseline…")
    kalman_results = run_kalman(DATASET, SPLIT)

    print("\nRunning linear extrapolation baseline…")
    linear_results = run_linear(DATASET, SPLIT)

    print("\nRunning mean trajectory baseline…")
    mean_results = run_mean_trajectory(DATASET, SPLIT)

    print("\nRunning model evaluation…")
    model_results = run_model(DATASET, CHECKPOINT_RUN, SPLIT, epoch=EPOCH, batch_size=BATCH_SIZE)

    output = {
        "higher_is_better": HIGHER_IS_BETTER,
        "kalman": kalman_results,
        "linear_extrapolation": linear_results,
        "mean_trajectory": mean_results,
        "model":  model_results,
    }

    os.makedirs("evaluation/results", exist_ok=True)
    out_path = f"evaluation/results/{CHECKPOINT_RUN}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved → {out_path}")
    print(f"Total evaluation time: {(time.time() - start_time)/60:.1f} minutes")
