"""metrics.py"""
import torch
import numpy as np
import torch.nn as nn

class FDEMetric():
    """
    Final Displacement Error (FDE).
    Euclidean distance between the predicted endpoint (argmax of heatmap)
    and the GT final position (last coordinate in coords).

    Cited: Gupta et al., "Social GAN", CVPR 2018.
    Gilles et al., "THOMAS", ICLR 2022 (heatmap-adapted FDE).

    Dummy definition:  How far off is your predicted endpoint?" You predict where the pedestrian will end up. 
                       GT says where they actually ended up. FDE measures the pixel distance between 
                       those two points. Small = you found the right spot
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


class MRMetric():
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

    Dummy definition:   "How often do you completely miss?" Same as FDE but binary — either 
                        you're close enough (within 20px) or you missed. MR is the percentage 
                        of complete misses. 0% = never miss, 100% = always miss.
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

class NearestGTPixelMetric():
    """
    Nearest GT Pixel Distance (NGP).
    Euclidean distance between the predicted endpoint (argmax of heatmap)
    and the nearest non-zero pixel in the ground truth raster.
    Unlike FDE which measures distance to the final GT point only,
    NGP rewards predictions that are close to ANY part of the future
    trajectory — better suited for blob-shaped spatial predictions.
    Lower = better.
    Cited: Inspired by endpoint-to-path distance evaluation in:
    Mangalam et al., "It Is Not the Journey but the Destination",
    ECCV 2020.

    Dummy definition:   "How far are you from any part of the correct path?" Instead of only checking the endpoint, 
                        this checks if your prediction lands anywhere near the GT trajectory. If the pedestrian walked 
                        a long path and you predicted a shorter one but in the right direction, NGP rewards you. 
                        FDE would punish you. Better suited for your blob predictions.
    """
    def forward(self, pred, target, coords):
        """
        pred:   (B, 1, H, W) — model output
        target: (B, 1, H, W) — GT raster
        coords: not used, kept for consistent interface
        """
        B = pred.shape[0]
        W = pred.shape[-1]
        ngp_vals = []
        for i in range(B):
            p = pred[i].squeeze().float()
            t = target[i].squeeze().float()

            # Predicted endpoint
            flat_idx = p.argmax()
            pred_y = (flat_idx // W).float()
            pred_x = (flat_idx  % W).float()

            # All non-zero GT pixels
            gt_pixels = (t > 0.01).nonzero(as_tuple=False).float()  # (N, 2) as (y, x)
            if gt_pixels.shape[0] == 0:
                continue

            gt_yx = gt_pixels  # (N, 2)
            pred_yx = torch.tensor([pred_y, pred_x], device=p.device).float()
            dists = torch.sqrt(((gt_yx - pred_yx) ** 2).sum(dim=1))
            ngp_vals.append(dists.min())

        return torch.stack(ngp_vals).mean() if ngp_vals else torch.tensor(0.0)


class DirectionalAccuracyMetric():
    """
    Directional Accuracy (DA).
    Cosine similarity between the predicted motion direction and the
    ground truth motion direction, both computed from coordinate sequences.
    GT direction: vector from first to last GT future coordinate.
    Predicted direction: vector from the GT current position (last past coord)
    to the predicted endpoint (argmax of heatmap).
    Range: [-1, 1] where 1 = perfect alignment, -1 = opposite direction.
    Higher = better.
    Cited: Similar directional evaluation used in:
    Salzmann et al., "Trajectron++", ECCV 2020.

    Dummy definition:   "Did you predict the right direction?" Draws a vector from where the pedestrian is now to
                        where you predicted they'll go. Draws another vector from where they are now to where 
                        they actually went. Measures how aligned those two vectors are. 1.0 = perfect direction,
                        0.0 = perpendicular, -1.0 = completely backwards.

    """
    def forward(self, pred, target, coords, past_coords):
        """
        pred:        (B, 1, H, W)
        coords:      (B, future_steps, 2) — future GT coordinates
        past_coords: (B, past_steps, 2)   — past coordinates, last = current pos
        """
        B = pred.shape[0]
        W = pred.shape[-1]
        da_vals = []
        for i in range(B):
            p = pred[i].squeeze().float()

            # Predicted direction: current pos → predicted endpoint
            flat_idx = p.argmax()
            pred_y = (flat_idx // W).float()
            pred_x = (flat_idx  % W).float()

            curr_x = past_coords[i, -1, 0].float()
            curr_y = past_coords[i, -1, 1].float()

            pred_dir = torch.stack([pred_x - curr_x, pred_y - curr_y])

            # GT direction: first → last future coordinate
            gt_start = coords[i, 0].float()
            gt_end   = coords[i, -1].float()
            gt_dir   = gt_end - gt_start

            # Cosine similarity
            norm_pred = pred_dir.norm() + 1e-8
            norm_gt   = gt_dir.norm() + 1e-8
            cosine    = (pred_dir * gt_dir).sum() / (norm_pred * norm_gt)
            da_vals.append(cosine)

        return torch.stack(da_vals).mean() if da_vals else torch.tensor(0.0)


class PathLengthRatioMetric():
    """
    Path Length Ratio (PLR).
    Ratio of predicted path length to GT path length.
    Predicted length: Euclidean distance from current position
    (last past coordinate) to predicted endpoint (argmax of heatmap).
    GT length: total arc length of the future coordinate sequence.
    PLR = predicted_length / gt_length
    A value of 1.0 indicates perfect length match.
    Values below 1.0 indicate underprediction (model predicts shorter
    trajectories than ground truth — expected due to perspective
    shortening in tilted-camera datasets and safe-blob collapse).
    Higher = better, but used primarily as a diagnostic metric to
    quantify the known length limitation of the model.
    This is a custom metric designed to explicitly measure the
    trajectory length limitation identified during experimentation.
    No external citation — proposed in this work.

    Dummy definition:   "How long is your prediction compared to reality?" GT pedestrian 
                        walked 100 pixels worth of path. You predicted they'd move 40 pixels. PLR = 0.4. 
                        This metric explicitly captures the known limitation — your model predicts shorter 
                        trajectories than reality due to perspective and the safe-blob tendency. Expected to be below 1.0,
                        and that's okay — it's there to be honest about the limitation, not to win.
    """
    def forward(self, pred, target, coords, past_coords):
        """
        pred:        (B, 1, H, W)
        coords:      (B, future_steps, 2)
        past_coords: (B, past_steps, 2)
        """
        B = pred.shape[0]
        W = pred.shape[-1]
        plr_vals = []
        for i in range(B):
            p = pred[i].squeeze().float()

            # Predicted length: current pos → argmax
            flat_idx = p.argmax()
            pred_y = (flat_idx // W).float()
            pred_x = (flat_idx  % W).float()

            curr_x = past_coords[i, -1, 0].float()
            curr_y = past_coords[i, -1, 1].float()

            pred_len = torch.sqrt((pred_x - curr_x)**2 + (pred_y - curr_y)**2)

            # GT length: arc length along future coords
            gt = coords[i].float()  # (T, 2)
            diffs = gt[1:] - gt[:-1]
            gt_len = torch.sqrt((diffs**2).sum(dim=1)).sum()

            if gt_len < 1e-3:
                continue  # skip stationary pedestrians

            plr_vals.append(pred_len / gt_len)

        return torch.stack(plr_vals).mean() if plr_vals else torch.tensor(0.0)