"""losses.py"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MAELoss(nn.Module):
    """Mean absolute error over the entire heatmap"""
    def forward(self, pred, target):
        return torch.abs(pred - target).mean()
    
class SparseHeatmapLoss(nn.Module): # this was best so far
    """
    Loss specifically for EXTREMELY sparse heatmaps (99.6% zeros).
    Combines:
    1. Standard MSE for overall structure
    2. Heavily weighted MSE for non-zero regions only
    3. L1 penalty to encourage sparsity in predictions
    """
    def __init__(self, nonzero_weight=100.0, sparsity_weight=0.1):
        super().__init__()
        self.nonzero_weight = nonzero_weight
        self.sparsity_weight = sparsity_weight
        
    def forward(self, pred, target):
        # 1. Basic MSE (small weight)
        mse_all = F.mse_loss(pred, target)
        
        # 2. MSE only on non-zero target regions (HEAVY weight)
        mask = (target > 0.01).float()
        n_nonzero = mask.sum() + 1e-8
        
        if n_nonzero > 1:
            mse_nonzero = ((pred - target) ** 2 * mask).sum() / n_nonzero
        else:
            mse_nonzero = torch.tensor(0.0, device=pred.device)
        
        # 3. L1 sparsity penalty (encourage pred to be sparse like target)
        # This penalizes the model for predicting non-zero everywhere
        sparsity_loss = pred.abs().mean()
        
        # Combine
        total_loss = (
            mse_all + 
            self.nonzero_weight * mse_nonzero + 
            self.sparsity_weight * sparsity_loss
        )
        
        return total_loss
    
class DiceLoss(nn.Module):
    """
    https://cvinvolution.medium.com/dice-loss-in-medical-image-segmentation-d0e476eb486
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs   = probs.view(probs.size(0), -1)     # (B, C, H, W) → (B, C*H*W)
        targets = targets.view(targets.size(0), -1) # (B, C, H, W) → (B, C*H*W)

        intersection = (probs*targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1-dice_coef.mean()
    

class NonZeroDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, threshold=0.5):
        super().__init__()
        self.smooth = smooth
        self.threshold = threshold

    def forward(self, pred, target):
        mask = (target > self.threshold).float()

        if mask.sum() == 0:
            # Return zero tensor with grad
            return torch.zeros(1, device=pred.device, dtype=pred.dtype, requires_grad=True)

        # Only select masked elements, keep them differentiable
        pred_masked = pred * mask
        target_masked = target * mask

        intersection = (pred_masked * target_masked).sum()
        union = pred_masked.sum() + target_masked.sum()

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class SparseIoULoss(nn.Module):
    """
    Soft IoU / Jaccard loss that focuses only on non-zero target pixels.
    Fully differentiable; ignores background pixels.
    """
    def __init__(self, target_threshold=0.5, smooth=1e-6):
        super().__init__()
        self.target_threshold = target_threshold
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred: [B,1,H,W] float in [0,1], model output after sigmoid
        target: [B,1,H,W] float in [0,1], sparse ground truth
        """
        # Mask for non-zero target pixels
        mask = (target > self.target_threshold).float()

        # If no non-zero pixels, return zero loss (or small constant)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Apply mask
        pred_masked = pred * mask
        target_masked = target * mask

        # Soft IoU
        intersection = (pred_masked * target_masked).sum()
        union = pred_masked.sum() + target_masked.sum() - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1.0 - iou

## From experiments
class KLDLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        B = pred.shape[0]
        kld_vals = []
        for i in range(B):
            p = pred[i].flatten().float()
            t = target[i].flatten().float()

            p = p / (p.sum() + self.eps)
            t = t / (t.sum() + self.eps)

            p = p + self.eps
            t = t + self.eps

            kld = (t * torch.log(t / p)).sum()
            kld_vals.append(kld)
        return torch.stack(kld_vals).mean()
    

class EMDLoss(nn.Module):
    def forward(self, pred, target):
        B = pred.shape[0]
        emd_vals = []
        for i in range(B):
            p = pred[i].flatten().float()
            t = target[i].flatten().float()

            p = p / (p.sum() + 1e-8)
            t = t / (t.sum() + 1e-8)

            p_cdf = torch.cumsum(p, dim=0)
            t_cdf = torch.cumsum(t, dim=0)

            emd_vals.append(torch.abs(p_cdf - t_cdf).mean())
        return torch.stack(emd_vals).mean()
    

class KLDEMDLoss(nn.Module):
    def __init__(self, kld_weight=1.0, emd_weight=1.0, eps=1e-8):
        super().__init__()
        self.kld_weight = float(kld_weight)
        self.emd_weight = float(emd_weight)
        self.eps = float(eps)

    def forward(self, pred, target):
        B = pred.shape[0]
        kld_vals, emd_vals = [], []
        for i in range(B):
            p = pred[i].flatten().float()
            t = target[i].flatten().float()

            p_n = p / (p.sum() + self.eps)
            t_n = t / (t.sum() + self.eps)

            # KLD
            kld_vals.append(((t_n + self.eps) * torch.log((t_n + self.eps) / (p_n + self.eps))).sum())

            # EMD
            p_cdf = torch.cumsum(p_n, dim=0)
            t_cdf = torch.cumsum(t_n, dim=0)
            emd_vals.append(torch.abs(p_cdf - t_cdf).mean())

        kld = torch.stack(kld_vals).mean()
        emd = torch.stack(emd_vals).mean()
        return self.kld_weight * kld + self.emd_weight * emd


class SparseHeatmapEMDLoss(nn.Module):
    def __init__(self, nonzero_weight=20.0, sparsity_weight=0.1, emd_weight=60.0):
        super().__init__()
        self.nonzero_weight = float(nonzero_weight)
        self.sparsity_weight = float(sparsity_weight)
        self.emd_weight = float(emd_weight)

    def forward(self, pred, target):
        # 1. Basic MSE
        mse_all = F.mse_loss(pred, target)

        # 2. MSE on non-zero regions
        mask = (target > 0.01).float()
        n_nonzero = mask.sum() + 1e-8
        if n_nonzero > 1:
            mse_nonzero = ((pred - target) ** 2 * mask).sum() / n_nonzero
        else:
            mse_nonzero = torch.tensor(0.0, device=pred.device)

        # 3. Sparsity penalty
        sparsity_loss = pred.abs().mean()

        # 4. EMD
        emd_vals = []
        for i in range(pred.shape[0]):
            p = pred[i].flatten().float()
            t = target[i].flatten().float()
            p = p / (p.sum() + 1e-8)
            t = t / (t.sum() + 1e-8)
            p_cdf = torch.cumsum(p, dim=0)
            t_cdf = torch.cumsum(t, dim=0)
            emd_vals.append(torch.abs(p_cdf - t_cdf).mean())
        emd = torch.stack(emd_vals).mean()

        return (
            mse_all +
            self.nonzero_weight * mse_nonzero +
            self.sparsity_weight * sparsity_loss +
            self.emd_weight * emd
        )


class FocalHeatmapLoss(nn.Module):
    """
    Pixel-wise focal loss from THOMAS (Gilles et al. 2022).
    L = -1/P * sum((Y - Y_hat)^2 * f(Y, Y_hat))
    where f = log(Y_hat)          if Y == 1
              (1-Y)^4 * log(1-Y_hat)  otherwise
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred, target):
        pred   = pred.float().clamp(self.eps, 1 - self.eps)
        target = target.float()

        # modulating factor (Y - Y_hat)^2
        mod = (target - pred) ** 2

        # focal weighting
        f_pos = torch.log(pred)                              # where Y == 1
        f_neg = (1 - target) ** 4 * torch.log(1 - pred)     # everywhere else

        # select based on whether pixel is exactly 1
        f = torch.where(target == 1.0, f_pos, f_neg)

        P = pred.numel()
        return -(mod * f).sum() / P
    
class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        pred_x   = F.conv2d(pred,   self.sobel_x, padding=1)
        pred_y   = F.conv2d(pred,   self.sobel_y, padding=1)
        target_x = F.conv2d(target, self.sobel_x, padding=1)
        target_y = F.conv2d(target, self.sobel_y, padding=1)

        pred_edge   = torch.sqrt(pred_x ** 2   + pred_y ** 2   + 1e-8)
        target_edge = torch.sqrt(target_x ** 2 + target_y ** 2 + 1e-8)

        return F.mse_loss(pred_edge, target_edge)
    
class FourierLoss(nn.Module):
    """
    Fourier loss from equation (13).
    Equivalent to Wasserstein W1 but computed in frequency domain.
    Penalizes high-frequency differences more than low-frequency ones
    via the 1/|k|^2 weighting.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        B = pred.shape[0]
        vals = []

        for i in range(B):
            p = pred[i].flatten().float()
            t = target[i].flatten().float()

            # normalize to probability distributions
            p = p / (p.sum() + self.eps)
            t = t / (t.sum() + self.eps)

            N = p.shape[0]

            # Fourier transforms
            p_hat = torch.fft.rfft(p)
            t_hat = torch.fft.rfft(t)

            # frequency indices k = 1, 2, ..., N//2
            k = torch.arange(1, p_hat.shape[0] + 1, device=p.device).float()

            # |mu_hat_k - nu_hat_k|^2 / |k|^2
            diff_sq = (p_hat - t_hat).abs() ** 2
            weighted = diff_sq / (k ** 2)

            vals.append(weighted.sum())

        return torch.stack(vals).mean()


class ActiveContourLoss(nn.Module):
    """
    Active Contour Loss based on Chan-Vese model.
    Originally from "Learning Active Contour Models for Medical Image Segmentation" (Chen et al., CVPR 2019).
    
    Region term: penalizes deviation from 0/1 values
    Length term: penalizes boundary length (encourages smooth shapes)
    """
    def __init__(self, length_weight=1.0, region_weight=1.0):
        super().__init__()
        self.length_weight = float(length_weight)
        self.region_weight = float(region_weight)

    def forward(self, pred, target):
        # --- Length term (boundary smoothness) ---
        # gradient in x and y directions
        dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]   # (B, 1, H, W-1)
        dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]   # (B, 1, H-1, W)

        length = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))

        # --- Region term ---
        # inside region: pred should match target (foreground)
        # outside region: pred should be 0 (background)
        inside  = torch.mean(target       * (1 - pred) ** 2)
        outside = torch.mean((1 - target) * pred ** 2)

        region = inside + outside

        return self.length_weight * length + self.region_weight * region

class SpatialMomentLoss(nn.Module):
    """
    Matches spatial moments between pred and target heatmaps:
    - 1st order: center of mass (x, y)
    - 2nd order: variance in x, variance in y, covariance (shape/orientation)
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = float(eps)

    def _moments(self, heatmap):
        B, _, H, W = heatmap.shape

        # normalize to sum to 1
        mass = heatmap.flatten(1).sum(dim=1, keepdim=True) + self.eps
        h    = heatmap / mass.view(B, 1, 1, 1)

        # coordinate grids
        ys = torch.linspace(0, 1, H, device=heatmap.device).view(1, 1, H, 1).expand(B, 1, H, W)
        xs = torch.linspace(0, 1, W, device=heatmap.device).view(1, 1, 1, W).expand(B, 1, H, W)

        # 1st order: mean position
        mean_x = (h * xs).flatten(1).sum(dim=1)  # (B,)
        mean_y = (h * ys).flatten(1).sum(dim=1)

        # 2nd order: variance and covariance
        var_x  = (h * (xs - mean_x.view(B, 1, 1, 1)) ** 2).flatten(1).sum(dim=1)
        var_y  = (h * (ys - mean_y.view(B, 1, 1, 1)) ** 2).flatten(1).sum(dim=1)
        cov_xy = (h * (xs - mean_x.view(B, 1, 1, 1)) * (ys - mean_y.view(B, 1, 1, 1))).flatten(1).sum(dim=1)

        return mean_x, mean_y, var_x, var_y, cov_xy

    def forward(self, pred, target):
        p_mx, p_my, p_vx, p_vy, p_cov = self._moments(pred)
        t_mx, t_my, t_vx, t_vy, t_cov = self._moments(target)

        loss = (
            F.mse_loss(p_mx,  t_mx)  +   # center x
            F.mse_loss(p_my,  t_my)  +   # center y
            F.mse_loss(p_vx,  t_vx)  +   # width
            F.mse_loss(p_vy,  t_vy)  +   # height
            F.mse_loss(p_cov, t_cov)      # orientation/skew
        )
        return loss
    

class SparseEMDEdgeLoss(nn.Module):
    """mačkopes číslo 1"""
    def __init__(self, nonzero_weight=100.0, sparsity_weight=0.1, emd_weight=130.0, edge_weight=580.0):
        super().__init__()
        self.nonzero_weight = float(nonzero_weight)
        self.sparsity_weight = float(sparsity_weight)
        self.emd_weight     = float(emd_weight)
        self.edge_weight    = float(edge_weight)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        # --- Sparse Heatmap ---
        mse_all      = F.mse_loss(pred, target)
        mask         = (target > 0.01).float()
        n_nonzero    = mask.sum() + 1e-8
        mse_nonzero  = ((pred - target) ** 2 * mask).sum() / n_nonzero if n_nonzero > 1 else torch.tensor(0.0, device=pred.device)
        sparsity     = pred.abs().mean()
        sparse       = mse_all + self.nonzero_weight * mse_nonzero + self.sparsity_weight * sparsity

        # --- EMD ---
        emd_vals = []
        for i in range(pred.shape[0]):
            p = pred[i].flatten().float()
            t = target[i].flatten().float()
            p = p / (p.sum() + 1e-8)
            t = t / (t.sum() + 1e-8)
            p_cdf = torch.cumsum(p, dim=0)
            t_cdf = torch.cumsum(t, dim=0)
            emd_vals.append(torch.abs(p_cdf - t_cdf).mean())
        emd = torch.stack(emd_vals).mean()

        # --- Edge ---
        sobel_x = self.sobel_x  # type: ignore
        sobel_y = self.sobel_y  # type: ignore
        pred_edge   = torch.sqrt(F.conv2d(pred,   sobel_x, padding=1) ** 2 + F.conv2d(pred,   sobel_y, padding=1) ** 2 + 1e-8)
        target_edge = torch.sqrt(F.conv2d(target, sobel_x, padding=1) ** 2 + F.conv2d(target, sobel_y, padding=1) ** 2 + 1e-8)
        edge = F.mse_loss(pred_edge, target_edge)

        return sparse + self.emd_weight * emd + self.edge_weight * edge
    
class FourierEdgeLoss(nn.Module):
    def __init__(self, fourier_weight=1.0, edge_weight=6.0, eps=1e-8):
        super().__init__()
        self.fourier_weight = float(fourier_weight)
        self.edge_weight    = float(edge_weight)
        self.eps            = float(eps)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        # --- Fourier ---
        vals = []
        for i in range(pred.shape[0]):
            p = pred[i].flatten().float()
            t = target[i].flatten().float()
            p = p / (p.sum() + self.eps)
            t = t / (t.sum() + self.eps)
            k       = torch.arange(1, torch.fft.rfft(p).shape[0] + 1, device=p.device).float()
            p_hat   = torch.fft.rfft(p)
            t_hat   = torch.fft.rfft(t)
            vals.append(((p_hat - t_hat).abs() ** 2 / k ** 2).sum())
        fourier = torch.stack(vals).mean()

        # --- Edge ---
        sobel_x = self.sobel_x  # type: ignore
        sobel_y = self.sobel_y  # type: ignore
        pred_edge   = torch.sqrt(F.conv2d(pred,   sobel_x, padding=1) ** 2 + F.conv2d(pred,   sobel_y, padding=1) ** 2 + 1e-8)
        target_edge = torch.sqrt(F.conv2d(target, sobel_x, padding=1) ** 2 + F.conv2d(target, sobel_y, padding=1) ** 2 + 1e-8)
        edge = F.mse_loss(pred_edge, target_edge)

        return self.fourier_weight * fourier + self.edge_weight * edge


class FourierEdgeSparseLoss(nn.Module):
    """Fourier + Edge + sparse nonzero term to fix trajectory truncation."""
    def __init__(self, fourier_weight=1.0, edge_weight=60.0, nonzero_weight=0.5, eps=1e-8):
        super().__init__()
        self.fourier_weight  = float(fourier_weight)
        self.edge_weight     = float(edge_weight)
        self.nonzero_weight  = float(nonzero_weight)
        self.eps             = float(eps)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        # --- Fourier ---
        vals = []
        for i in range(pred.shape[0]):
            p = pred[i].flatten().float()
            t = target[i].flatten().float()
            p = p / (p.sum() + self.eps)
            t = t / (t.sum() + self.eps)
            p_hat = torch.fft.rfft(p)
            t_hat = torch.fft.rfft(t)
            k     = torch.arange(1, p_hat.shape[0] + 1, device=p.device).float()
            vals.append(((p_hat - t_hat).abs() ** 2 / k ** 2).sum())
        fourier = torch.stack(vals).mean()

        # --- Edge ---
        sobel_x = self.sobel_x  # type: ignore
        sobel_y = self.sobel_y  # type: ignore
        pred_edge   = torch.sqrt(F.conv2d(pred,   sobel_x, padding=1) ** 2 + F.conv2d(pred,   sobel_y, padding=1) ** 2 + 1e-8)
        target_edge = torch.sqrt(F.conv2d(target, sobel_x, padding=1) ** 2 + F.conv2d(target, sobel_y, padding=1) ** 2 + 1e-8)
        edge = F.mse_loss(pred_edge, target_edge)

        # --- Sparse nonzero term ---
        mask        = (target > 0.01).float()
        n_nonzero   = mask.sum() + 1e-8
        mse_nonzero = ((pred - target) ** 2 * mask).sum() / n_nonzero

        return self.fourier_weight * fourier + self.edge_weight * edge + self.nonzero_weight * mse_nonzero


class SparseEMDEdgeCoverageLoss(nn.Module):
    """Sparse + EMD + Edge + coverage term to fix trajectory truncation."""
    def __init__(self, nonzero_weight=10.0, sparsity_weight=0.01, emd_weight=2.0, edge_weight=60.0, coverage_weight=5.0):
        super().__init__()
        self.nonzero_weight  = float(nonzero_weight)
        self.sparsity_weight = float(sparsity_weight)
        self.emd_weight      = float(emd_weight)
        self.edge_weight     = float(edge_weight)
        self.coverage_weight = float(coverage_weight)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        # --- Sparse Heatmap ---
        mse_all     = F.mse_loss(pred, target)
        mask        = (target > 0.01).float()
        n_nonzero   = mask.sum() + 1e-8
        mse_nonzero = ((pred - target) ** 2 * mask).sum() / n_nonzero if n_nonzero > 1 else torch.tensor(0.0, device=pred.device)
        sparsity    = pred.abs().mean()
        sparse      = mse_all + self.nonzero_weight * mse_nonzero + self.sparsity_weight * sparsity

        # --- EMD ---
        emd_vals = []
        for i in range(pred.shape[0]):
            p = pred[i].flatten().float()
            t = target[i].flatten().float()
            p = p / (p.sum() + 1e-8)
            t = t / (t.sum() + 1e-8)
            p_cdf = torch.cumsum(p, dim=0)
            t_cdf = torch.cumsum(t, dim=0)
            emd_vals.append(torch.abs(p_cdf - t_cdf).mean())
        emd = torch.stack(emd_vals).mean()

        # --- Edge ---
        sobel_x = self.sobel_x  # type: ignore
        sobel_y = self.sobel_y  # type: ignore
        pred_edge   = torch.sqrt(F.conv2d(pred,   sobel_x, padding=1) ** 2 + F.conv2d(pred,   sobel_y, padding=1) ** 2 + 1e-8)
        target_edge = torch.sqrt(F.conv2d(target, sobel_x, padding=1) ** 2 + F.conv2d(target, sobel_y, padding=1) ** 2 + 1e-8)
        edge = F.mse_loss(pred_edge, target_edge)

        # --- Coverage ---
        pred_coverage = (pred   > 0.05).float().sum()
        gt_coverage   = (target > 0.05).float().sum()
        coverage      = F.relu(gt_coverage - pred_coverage) / (gt_coverage + 1e-8)

        return sparse + self.emd_weight * emd + self.edge_weight * edge + self.coverage_weight * coverage
    
class SparseEMDLoss(nn.Module):
    def __init__(self, nonzero_weight=100.0, sparsity_weight=0.1, emd_weight=1.0, sparse_weight=0.004, eps=1e-8):
        super().__init__()
        self.nonzero_weight  = float(nonzero_weight)
        self.sparsity_weight = float(sparsity_weight)
        self.emd_weight      = float(emd_weight)
        self.sparse_weight   = float(sparse_weight)  # 0.015×0.004 ≈ 0.004 → sparse ~25% of emd
        self.eps             = float(eps)

    def forward(self, pred, target):
        # --- Sparse Heatmap ---
        mse_all     = F.mse_loss(pred, target)
        mask        = (target > 0.01).float()
        n_nonzero   = mask.sum() + 1e-8
        mse_nonzero = ((pred - target) ** 2 * mask).sum() / n_nonzero if n_nonzero > 1 else torch.tensor(0.0, device=pred.device)
        sparsity    = pred.abs().mean()
        sparse      = mse_all + self.nonzero_weight * mse_nonzero + self.sparsity_weight * sparsity

        # --- EMD ---
        emd_vals = []
        for i in range(pred.shape[0]):
            p = pred[i].flatten().float()
            t = target[i].flatten().float()
            p = p / (p.sum() + self.eps)
            t = t / (t.sum() + self.eps)
            p_cdf = torch.cumsum(p, dim=0)
            t_cdf = torch.cumsum(t, dim=0)
            emd_vals.append(torch.abs(p_cdf - t_cdf).mean())
        emd = torch.stack(emd_vals).mean()

        return self.emd_weight * emd + self.sparse_weight * sparse


class EMDFourierLoss(nn.Module):
    def __init__(self, emd_weight=31.0, fourier_weight=1.0, eps=1e-8):
        super().__init__()
        self.emd_weight     = float(emd_weight)
        self.fourier_weight = float(fourier_weight)
        self.eps            = float(eps)

    def forward(self, pred, target):
        # --- EMD ---
        emd_vals = []
        for i in range(pred.shape[0]):
            p = pred[i].flatten().float()
            t = target[i].flatten().float()
            p = p / (p.sum() + self.eps)
            t = t / (t.sum() + self.eps)
            p_cdf = torch.cumsum(p, dim=0)
            t_cdf = torch.cumsum(t, dim=0)
            emd_vals.append(torch.abs(p_cdf - t_cdf).mean())
        emd = torch.stack(emd_vals).mean()

        # --- Fourier ---
        fourier_vals = []
        for i in range(pred.shape[0]):
            p = pred[i].flatten().float()
            t = target[i].flatten().float()
            p = p / (p.sum() + self.eps)
            t = t / (t.sum() + self.eps)
            p_hat = torch.fft.rfft(p)
            t_hat = torch.fft.rfft(t)
            k     = torch.arange(1, p_hat.shape[0] + 1, device=p.device).float()
            fourier_vals.append(((p_hat - t_hat).abs() ** 2 / k ** 2).sum())
        fourier = torch.stack(fourier_vals).mean()

        return self.emd_weight * emd + self.fourier_weight * fourier