"""losses.py"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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
    
class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.c = omega - omega * np.log(1 + omega / epsilon)
    
    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1 + diff / self.epsilon),
            diff - self.c
        )
        return loss.mean()
    

class FocalMSELoss(nn.Module):
    """
    Focal MSE that heavily weights the sparse non-zero regions.
    Perfect for extremely sparse heatmaps (99.63% zeros).
    """
    def __init__(self, alpha=10.0, beta=2.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # weight for non-zero regions
        self.beta = beta    # weight for MSE component
        self.gamma = gamma  # focal weight exponent
        
    def forward(self, pred, target):
        # Binary mask for non-zero regions
        mask = (target > 0.01).float()  # your min non-zero is 0.003922
        
        # MSE components
        mse_all = F.mse_loss(pred, target, reduction='none')
        mse_nonzero = mse_all * mask
        
        # Focal weighting (harder examples get more weight)
        focal_weight = torch.abs(pred - target) ** self.gamma
        
        # Combine losses
        loss_all = (mse_all * focal_weight).mean()
        loss_nonzero = (mse_nonzero * focal_weight).mean()
        
        # Heavily weight the non-zero regions
        total_loss = self.beta * loss_all + self.alpha * loss_nonzero
        
        return total_loss


class WeightedMSELoss(nn.Module):
    """
    Simpler alternative: just heavily weight non-zero pixels.
    """
    def __init__(self, zero_weight=1.0, nonzero_weight=100.0):
        super().__init__()
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight
        
    def forward(self, pred, target):
        # Create weight map
        weights = torch.ones_like(target) * self.zero_weight
        weights[target > 0.01] = self.nonzero_weight
        
        # Weighted MSE
        mse = (pred - target) ** 2
        weighted_mse = mse * weights
        
        return weighted_mse.mean()
    
class SparseHeatmapLoss(nn.Module): # this was so far cool
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
    """
    Dice Loss computed ONLY on non-zero target regions.
    Ignores background pixels completely.
    Good for sparse heatmaps where background dominates.
    """
    def __init__(self, smooth=1e-6, threshold=0.01):
        super().__init__()
        self.smooth = smooth
        self.threshold = threshold  # min value to consider "non-zero"
        
    def forward(self, pred, target):
        # Create mask for non-zero regions
        mask = (target > self.threshold).float()
        
        # Apply mask to both pred and target
        pred_masked = pred * mask
        target_masked = target * mask
        
        # Flatten
        pred_flat = pred_masked.view(-1)
        target_flat = target_masked.view(-1)
        
        # Dice coefficient on masked regions only
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice