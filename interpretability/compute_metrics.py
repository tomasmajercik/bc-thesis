"""
compute_metrics.py

Load a checkpoint, run a full forward pass over the validation set,
and compute all 5 metrics. Results are saved to:
    previews/metrics/{checkpoint-name}.json

Usage:
    python -m interpretability.compute_metrics --checkpoint checkpoints/my-run/best_model.pth
    python -m interpretability.compute_metrics --checkpoint checkpoints/my-run/best_model.pth --split val
    python -m interpretability.compute_metrics --checkpoint checkpoints/my-run/best_model.pth --split full
"""

import os
import json
import argparse

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from model.model import MultiEncoderUNet
from training.datasets import PETSDataset, PETSDatasetLT, PETSDatasetST, PETSDatasetSW, PETSDatasetLW, PETS09_Old
from training.utils import split_ds_sequential
from training.metrics import EMDMetric, KLDMetric, NSSMetric, FDEMetric, MRMetric, NLLMetric, TopKCoverageMetric

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIGHER_IS_BETTER = {
    "EMD":  False,
    "KLD":  False,
    "NSS":  True,
    "FDE":  False,
    "MR":   False,
    "NLL":  False,
    "TOPk": True,
}


def load_model(ckpt_path: str) -> MultiEncoderUNet:
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "best_model.pth")
    model = MultiEncoderUNet().to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def compute(ckpt_path: str, split: str = "val", batch_size: int = 8):
    dataset = PETS09_Old(scale=0.5, return_coords=True)
    # dataset = PETSDatasetLT(scale=0.5, return_coords=True)
    # dataset = PETSDatasetST(scale=0.5, return_coords=True)
    # dataset = PETSDatasetSW(scale=0.5, return_coords=True)
    # dataset = PETSDatasetLW(scale=0.5, return_coords=True)

    if split == "full":
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        train_ds, val_ds, test_ds = split_ds_sequential(dataset, 0.7, 0.2)
        ds = val_ds if split == "val" else train_ds
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = load_model(ckpt_path)

    metrics = {
        "EMD":  EMDMetric(),
        "KLD":  KLDMetric(),
        "NSS":  NSSMetric(),
        "FDE":  FDEMetric(),
        "MR":   MRMetric(threshold_px=20.0),
        "NLL":  NLLMetric(),
    }

    accumulators = {name: [] for name in metrics}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            past, obstacle, context, zoom, target, coords = [x.to(DEVICE) for x in batch]
            pred = model(past, obstacle, context, zoom)

            for name, metric_fn in metrics.items():
                accumulators[name].append(metric_fn(pred, target, coords).item())

    results = {name: float(sum(vals) / len(vals)) for name, vals in accumulators.items()}

    # Save
    if os.path.isdir(ckpt_path):
        ckpt_name = os.path.basename(ckpt_path.rstrip("/"))
    else:
        ckpt_name = os.path.basename(os.path.dirname(ckpt_path))
    out_dir = "previews/metrics"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ckpt_name}.json")

    with open(out_path, "w") as f:
        json.dump({**results, "higher_is_better": HIGHER_IS_BETTER}, f, indent=2)

    print(f"\nSaved → {out_path}")
    for name, val in results.items():
        print(f"  {name}: {val:.6f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--split", default="val", choices=["train", "val", "full"],
                        help="Which split to evaluate on (default: val)")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    compute(args.checkpoint, split=args.split, batch_size=args.batch_size)

# python -m interpretability.compute_metrics --checkpoint <path>