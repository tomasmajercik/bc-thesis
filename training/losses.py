"""losses.py"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MAELoss(nn.Module):
    """Mean absolute error over the entire heatmap"""
    def forward(self, pred, target):
        return torch.abs(pred - target).mean()
    
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
