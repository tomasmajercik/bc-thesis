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
    
class EdgeCoverageLoss(nn.Module):
    def __init__(self, coverage_weight=0.1, threshold=0.2):
        super().__init__()
        self.coverage_weight = float(coverage_weight)
        self.threshold       = float(threshold)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        sobel_x = self.sobel_x  # type: ignore
        sobel_y = self.sobel_y  # type: ignore

        # --- Edge ---
        pred_edge   = torch.sqrt(F.conv2d(pred,   sobel_x, padding=1) ** 2 + F.conv2d(pred,   sobel_y, padding=1) ** 2 + 1e-8)
        target_edge = torch.sqrt(F.conv2d(target, sobel_x, padding=1) ** 2 + F.conv2d(target, sobel_y, padding=1) ** 2 + 1e-8)
        edge = F.mse_loss(pred_edge, target_edge)

        # --- Soft IoU coverage ---
        # only penalize pred for NOT covering GT regions
        # does not reward blobs outside GT
        gt_mask   = (target > self.threshold).float()
        pred_soft = pred * gt_mask          # pred values INSIDE gt region only
        gt_soft   = target * gt_mask

        intersection = (pred_soft * gt_soft).sum(dim=[1, 2, 3])
        gt_area      = gt_soft.sum(dim=[1, 2, 3]) + 1e-8
        coverage     = 1.0 - (intersection / gt_area).mean()  # 0 = perfect, 1 = no coverage

        # print(f"Edge: {edge.item():.4f}  Coverage: {self.coverage_weight * coverage:.4f}")
        return edge + self.coverage_weight * coverage
    
class EdgeSparseLoss(nn.Module): # edgeloss but should force the model to stretch the lines
    def __init__(self, edge_weight=1.0, nonzero_weight=0.5):
        super().__init__()
        self.edge_weight    = float(edge_weight)
        self.nonzero_weight = float(nonzero_weight)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, pred, target):
        # --- Edge ---
        sobel_x = self.sobel_x  # type: ignore
        sobel_y = self.sobel_y  # type: ignore
        pred_edge   = torch.sqrt(F.conv2d(pred,   sobel_x, padding=1) ** 2 + F.conv2d(pred,   sobel_y, padding=1) ** 2 + 1e-8)
        target_edge = torch.sqrt(F.conv2d(target, sobel_x, padding=1) ** 2 + F.conv2d(target, sobel_y, padding=1) ** 2 + 1e-8)
        edge = F.mse_loss(pred_edge, target_edge)

        # --- Nonzero region MSE (forces line extent) ---
        mask        = (target > 0.01).float()
        n_nonzero   = mask.sum() + 1e-8
        mse_nonzero = ((pred - target) ** 2 * mask).sum() / n_nonzero

        return self.edge_weight * edge + self.nonzero_weight * mse_nonzero

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
    

class ChamferHeatmapLoss(nn.Module):
    """
    Soft Chamfer loss between predicted and GT heatmaps.
    Treats heatmaps as weighted point clouds.
    Penalizes predictions that are blob-shaped when GT is elongated,
    and predictions that miss the trajectory endpoints.
    """
    def __init__(self, top_k_frac=0.05):
        super().__init__()
        self.top_k_frac = top_k_frac  # fraction of pixels to treat as "active"

    def forward(self, pred, target):
        B, _, H, W = pred.shape
        device = pred.device
        eps = 1e-8
        losses = []

        # Pixel coordinate grid — fixed geometry, no gradients needed
        ys = torch.arange(H, device=device).float()
        xs = torch.arange(W, device=device).float()
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        all_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (N, 2)

        for i in range(B):
            p = pred[i].squeeze().flatten()    # (N,) — requires grad
            t = target[i].squeeze().flatten()  # (N,) — no grad needed

            # Soft distribution over pred pixels — gradient flows through here
            p_norm = p / (p.sum() + eps)

            # Hard-select top-k GT pixels (GT has no grad anyway)
            k = max(1, int(self.top_k_frac * H * W))
            _, gt_idx = torch.topk(t, k)
            gt_pts = all_coords[gt_idx]  # (k, 2)

            # Distance from every pixel to its nearest GT point: (N,)
            # This is pure geometry — a fixed weight map, no grad needed
            dist_to_gt = torch.cdist(all_coords, gt_pts).min(dim=1).values  # (N,)

            # pred → GT: pull pred mass toward GT pixels (grad flows through p_norm)
            loss_p2g = (p_norm * dist_to_gt).sum()

            # GT → pred: expected distance from each GT point under pred distribution
            # dist_all_to_gt.t() is (k, N): dist from each GT pt to every pred pixel
            dist_gt_to_all = torch.cdist(gt_pts, all_coords)          # (k, N)
            expected_dist = (dist_gt_to_all * p_norm.unsqueeze(0)).sum(dim=1)  # (k,)
            loss_g2p = expected_dist.mean()

            losses.append(loss_p2g + loss_g2p)

        return torch.stack(losses).mean()
    
class TverskyLoss(nn.Module):
    """
    Tversky Loss for sparse heatmap prediction.
    Generalizes Dice loss with asymmetric penalties for FP and FN.
    
    alpha: penalty for false positives (predicting trajectory where there is none)
    beta:  penalty for false negatives (missing trajectory that exists)
    
    Set beta > alpha to punish missing the trajectory more than over-predicting.
    Recommended starting point: alpha=0.3, beta=0.7
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.8, eps: float = 1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred, target):
        # Both pred and target are already in [0, 1] — no per-sample max normalization.
        # Per-max normalization was the instability source: a short high-intensity blob
        # and a long dim line became identical after dividing by their own maxes,
        # destroying the FN asymmetry and causing erratic gradients early in training.
        p = pred.flatten(1).float()
        t = target.flatten(1).float()

        tp = (p * t).sum(dim=1)
        fp = (p * (1 - t)).sum(dim=1)
        fn = ((1 - p) * t).sum(dim=1)

        tversky = tp / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return (1 - tversky).mean()


class RecallWithToleranceLoss(nn.Module):
    """
    Directly solves the short-line problem by separating recall from precision
    with asymmetric weights and spatial tolerance.

    Core idea
    ---------
    For every GT pixel, ask: "does the prediction have *any* activation
    within `tolerance_px` pixels of this location?"  If yes, no recall
    penalty.  If no, heavy penalty.  Precision is penalised only for
    predictions that are far from *every* GT pixel — so a slightly off-axis
    but full-length line is virtually free.

    Why this works where others failed
    -----------------------------------
    - No sparsity term  → the model is never rewarded for being short.
    - Tolerance window  → imprecise-but-long predictions are accepted.
    - recall_weight >> precision_weight → missing the far end of the
      trajectory hurts far more than predicting a few extra pixels.
    - Max-pool trick    → the tolerance is implemented in O(HW) via
      max-pooling instead of expensive pairwise distances.

    Parameters
    ----------
    tolerance_px    : spatial slack in pixels (GT is soft-dilated by this amount).
                      Start with 5–7 px; increase if the trajectory is thick.
    recall_weight   : how much missing a GT pixel costs.  >= 10 recommended.
    precision_weight: how much predicting far outside GT costs.  Keep small
                      (0.1–0.5) — you want imprecise predictions to be cheap.
    threshold       : minimum GT value treated as "active" (ignores near-zero noise).
    """

    def __init__(
        self,
        tolerance_px: int    = 5,
        recall_weight: float  = 15.0,
        precision_weight: float = 0.3,
        threshold: float     = 0.05,
    ):
        super().__init__()
        self.tolerance_px     = int(tolerance_px)
        self.recall_weight    = float(recall_weight)
        self.precision_weight = float(precision_weight)
        self.threshold        = float(threshold)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        k = 2 * self.tolerance_px + 1

        # --- RECALL ---
        # For every GT pixel, take the maximum prediction within a window.
        # If that max >= GT value the pixel is "covered" → zero recall penalty.
        pred_dilated = F.max_pool2d(
            pred,
            kernel_size=k,
            stride=1,
            padding=self.tolerance_px,
        )  # each location holds the max pred in its neighbourhood

        gt_mask     = (target > self.threshold).float()
        # relu: only penalise when gt > pred_dilated (missed GT, not over-predicted)
        recall_loss = (gt_mask * F.relu(target - pred_dilated)).mean()

        # --- PRECISION ---
        # Only penalise predictions that are far from any GT pixel.
        gt_dilated    = F.max_pool2d(
            gt_mask,
            kernel_size=k,
            stride=1,
            padding=self.tolerance_px,
        )
        outside_zone   = 1.0 - gt_dilated           # 1 = far from any GT pixel
        precision_loss = (pred * outside_zone).pow(2).mean()

        return self.recall_weight * recall_loss + self.precision_weight * precision_loss