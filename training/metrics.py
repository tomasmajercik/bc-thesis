"""metrics.py"""
import torch
import numpy as np
import torch.nn as nn


# ==============================================================================
# Base classes (Option A: unified signature with optional coords)
# ==============================================================================

class HeatmapMetric(nn.Module):
    """
    Base class for metrics that operate on heatmap tensors only.
    forward(pred, target, coords=None) — coords ignored.
    """
    def forward(self, pred, target, coords=None):
        raise NotImplementedError


class CoordMetric(nn.Module):
    """
    Base class for metrics that require raw GT coordinates.
    forward(pred, target, coords) — target heatmap may be ignored.
    coords: (B, future_steps, 2) float tensor of GT pixel positions (x, y).
    """
    def forward(self, pred, target, coords):
        raise NotImplementedError


# ==============================================================================
# Heatmap metrics
# ==============================================================================

class EMDMetric(HeatmapMetric):
    """
    Earth Mover's Distance (Wasserstein-1) between predicted and GT heatmaps.
    Both maps are normalized to sum to 1 before comparison.
    Uses the 1D sliced approximation over flattened maps for efficiency.

    Cited: Bylinskii et al., "What Do Different Evaluation Metrics Tell Us
    About Saliency Models?", IEEE TPAMI 2019.
    """
    def forward(self, pred, target, coords=None):
        B = pred.shape[0]
        emd_vals = []

        for i in range(B):
            p = pred[i].flatten().float()
            t = target[i].flatten().float()

            # Normalize to valid distributions
            p = p / (p.sum() + 1e-8)
            t = t / (t.sum() + 1e-8)

            # Wasserstein-1 via CDF difference on sorted 1D projection
            p_cdf = torch.cumsum(p, dim=0)
            t_cdf = torch.cumsum(t, dim=0)

            emd_vals.append(torch.abs(p_cdf - t_cdf).mean())

        return torch.stack(emd_vals).mean()


class KLDMetric(HeatmapMetric):
    """
    KL Divergence KL(GT || pred) between predicted and GT heatmaps.
    Both maps normalized to sum to 1. Epsilon smoothing avoids log(0).
    Convention KL(GT||pred): penalizes pred for missing GT mass.

    Cited: Bylinskii et al., "What Do Different Evaluation Metrics Tell Us
    About Saliency Models?", IEEE TPAMI 2019.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, coords=None):
        B = pred.shape[0]
        kld_vals = []

        for i in range(B):
            p = pred[i].flatten().float()
            t = target[i].flatten().float()

            p = p / (p.sum() + self.eps)
            t = t / (t.sum() + self.eps)

            # Smooth to avoid log(0)
            p = p + self.eps
            t = t + self.eps

            kld = (t * torch.log(t / p)).sum()
            kld_vals.append(kld)

        return torch.stack(kld_vals).mean()


class NSSMetric(CoordMetric):
    """
    Normalized Scanpath Saliency (NSS).
    Measures the mean normalized predicted saliency at GT fixation locations.
    Higher = better. NSS=0 means prediction is at chance.

    GT coords are used as fixation points (no thresholding needed).

    Cited: Peters et al., "Components of bottom-up gaze allocation in natural
    images", Vision Research 2005. Used in saliency benchmarks by
    Bylinskii et al., IEEE TPAMI 2019.
    """
    def forward(self, pred, target, coords):
        """
        coords: (B, future_steps, 2) — pixel positions (x, y) in heatmap space
        """
        B = pred.shape[0]
        H = pred.shape[-2]
        W = pred.shape[-1]
        nss_vals = []

        for i in range(B):
            p = pred[i].squeeze().float()           # (H, W)

            # Normalize prediction: zero mean, unit std
            p_norm = (p - p.mean()) / (p.std() + 1e-8)

            # Collect NSS values at each GT fixation point
            sample_coords = coords[i]               # (future_steps, 2)
            scores = []

            for (cx, cy) in sample_coords:
                px = int(cx.item())
                py = int(cy.item())

                # Skip out-of-bounds coords
                if 0 <= px < W and 0 <= py < H:
                    scores.append(p_norm[py, px])

            if len(scores) == 0:
                nss_vals.append(torch.tensor(0.0, device=pred.device))
            else:
                nss_vals.append(torch.stack(scores).mean())

        return torch.stack(nss_vals).mean()


# ==============================================================================
# Coordinate metrics
# ==============================================================================

class FDEMetric(CoordMetric):
    """
    Final Displacement Error (FDE).
    Euclidean distance between the predicted endpoint (argmax of heatmap)
    and the GT final position (last coordinate in coords).

    Cited: Gupta et al., "Social GAN", CVPR 2018.
    Gilles et al., "THOMAS", ICLR 2022 (heatmap-adapted FDE).
    """
    def forward(self, pred, target, coords):
        """
        coords: (B, future_steps, 2) — last entry is the final GT position
        """
        B = pred.shape[0]
        W = pred.shape[-1]
        fde_vals = []

        for i in range(B):
            p = pred[i].squeeze().float()           # (H, W)

            # Predicted endpoint: argmax of heatmap
            flat_idx = p.argmax()
            pred_y = (flat_idx // W).float()
            pred_x = (flat_idx  % W).float()

            # GT endpoint: last future coordinate
            gt_x, gt_y = coords[i, -1, 0].float(), coords[i, -1, 1].float()

            fde = torch.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            fde_vals.append(fde)

        return torch.stack(fde_vals).mean()


class MRMetric(CoordMetric):
    """
    Miss Rate (MR).
    Fraction of samples where the predicted endpoint (argmax of heatmap)
    falls outside a radius threshold from the GT final position.
    Lower = better.

    Threshold of 1m ≈ common in trajectory prediction literature.
    In pixel space, set threshold according to your dataset's px/m ratio.

    Cited: Ettinger et al., "Large Scale Interactive Motion Forecasting
    for Autonomous Driving", ICCV 2021.
    Gilles et al., "GOHOME", ICRA 2022.
    """
    def __init__(self, threshold_px: float = 20.0):
        super().__init__()
        self.threshold_px = threshold_px

    def forward(self, pred, target, coords):
        """
        coords: (B, future_steps, 2) — last entry is the final GT position
        """
        B = pred.shape[0]
        W = pred.shape[-1]
        misses = []

        for i in range(B):
            p = pred[i].squeeze().float()

            flat_idx = p.argmax()
            pred_y = (flat_idx // W).float()
            pred_x = (flat_idx  % W).float()

            gt_x, gt_y = coords[i, -1, 0].float(), coords[i, -1, 1].float()

            dist = torch.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            misses.append((dist > self.threshold_px).float())

        return torch.stack(misses).mean()
