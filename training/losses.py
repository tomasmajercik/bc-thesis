import torch
import torch.nn as nn

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