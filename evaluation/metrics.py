""" this file contains metrics used in the final evaluation.
Each metric is implemented hera and has short docstrings with definitions"""
import torch

class MRMetric():
    """
    Miss Rate (MR).
    the fraction of samples in which the predicted endpoint falls outside a 
    fixed pixel radius from the final ground truth position, penalizing
    predictions with large endpoint deviations.

    Dummy definition:   What fraction of predictions completely miss the target? 0 = never miss, 1 = always miss.
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


class FDEMetric():
    """
    Final Displacement Error (FDE).
    Euclidean distance between the predicted endpoint and the final ground truth position.

    Dummy definition:  Average pixel distance between predicted endpoint and actual endpoint. Lower = more accurate.
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
    



class DirectionalAccuracyMetric():
    """
    Directional Accuracy (DA).
    cosine similarity between the predicted motion direction and the ground truth motion
    direction, rewarding correct motion direction regardless of magnitude

    Dummy definition:   How well does the predicted direction align with actual direction? 1.0 = perfect, 0.0 = perpendicular, -1.0 = opposite

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

            flat_idx = p.argmax()
            pred_y = (flat_idx // W).float()
            pred_x = (flat_idx  % W).float()

            pc = past_coords[i]
            valid_pc = pc[pc[:, 0] >= 0]
            if valid_pc.shape[0] == 0:
                continue
            curr_x = valid_pc[-1, 0].float()
            curr_y = valid_pc[-1, 1].float()

            pred_dir = torch.stack([pred_x - curr_x, -(pred_y - curr_y)]) # Invert y-axis to match typical Cartesian coordinates where up is positive

            # GT direction: first → last future coordinate (filter -1 padding)
            valid_fc = coords[i][coords[i, :, 0] >= 0]
            if valid_fc.shape[0] < 2:
                continue
            gt_start = valid_fc[0].float()
            gt_end   = valid_fc[-1].float()
            gt_dir   = gt_end - gt_start

            # Cosine similarity
            norm_pred = pred_dir.norm() + 1e-8
            norm_gt   = gt_dir.norm() + 1e-8
            cosine    = (pred_dir * gt_dir).sum() / (norm_pred * norm_gt)
            da_vals.append(cosine)

        return torch.stack(da_vals).mean() if da_vals else torch.tensor(0.0)
    
class PathCoverageMetric():
    """
    Path Coverage @ K (PC@K).
    the fraction of ground truth future positions that are covered by the prediction,
    where a position is considered covered if the predicted output has
    a value above a fixed threshold within a given pixel radius

    Dummy definition:   What fraction of the pedestrian's future path is covered by the prediction? 1.0 = full path covered, 0.0 = nothing covered.
    """
    def __init__(self, radius_px: float = 20.0, threshold: float = 0.1):
        self.radius_px = radius_px
        self.threshold = threshold

    def forward(self, pred, target, coords):
        """
        pred:   (B, 1, H, W) — model output in [0, 1]
        target: (B, 1, H, W) — not used, kept for consistent interface
        coords: (B, future_steps, 2) — GT future positions (x, y)
        """
        B = pred.shape[0]
        H = pred.shape[2]
        W = pred.shape[3]
        coverage_vals = []

        for i in range(B):
            p = pred[i].squeeze().float()  # (H, W)

            # precompute coordinate grids once
            ys = torch.arange(H, device=p.device).float()
            xs = torch.arange(W, device=p.device).float()
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)

            covered = 0
            total = 0

            for step in range(coords.shape[1]):
                gt_x = coords[i, step, 0].float()
                gt_y = coords[i, step, 1].float()

                # skip padded invalid coords
                if gt_x < 0 or gt_y < 0:
                    continue

                total += 1

                # find max prediction within radius_px of this GT point
                dist_map = torch.sqrt((grid_x - gt_x)**2 + (grid_y - gt_y)**2)
                within_radius = dist_map <= self.radius_px
                max_pred_nearby = p[within_radius].max() if within_radius.any() else torch.tensor(0.0)

                if max_pred_nearby >= self.threshold:
                    covered += 1

            if total > 0:
                coverage_vals.append(torch.tensor(covered / total))

        return torch.stack(coverage_vals).mean() if coverage_vals else torch.tensor(0.0)

class PathLengthRatioMetric():
    """
    Path Length Ratio (PLR).
    the ratio of predicted trajectory length to ground truth trajectory length

    Dummy definition:   Ratio of predicted trajectory length to ground truth length. 1.0 = same length, below 1.0 = shorter prediction, above 1.0 = longer prediction.
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

            # past_coords may be padded with -1 up to no_steps; take last valid entry
            pc = past_coords[i]
            valid_pc = pc[pc[:, 0] >= 0]
            if valid_pc.shape[0] == 0:
                continue
            curr_x = valid_pc[-1, 0].float()
            curr_y = valid_pc[-1, 1].float()

            pred_len = torch.sqrt((pred_x - curr_x)**2 + (pred_y - curr_y)**2)

            # GT length: arc length along valid (non-padded) future coords
            gt = coords[i][coords[i, :, 0] >= 0].float()
            diffs = gt[1:] - gt[:-1]
            gt_len = torch.sqrt((diffs**2).sum(dim=1)).sum()

            if gt_len < 1e-3:
                continue  # skip stationary pedestrians

            plr_vals.append(pred_len / gt_len)

        return torch.stack(plr_vals).mean() if plr_vals else torch.tensor(0.0)

## This metric was tested but not used in the final evaluation
class NearestGTPixelMetric():
    """
    Euclidean distance from the predicted endpoint to the nearest non-zero 
    pixel in the ground truth raster, rewarding predictions that are close
    to any part of the future trajectory rather than only the endpoint.
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