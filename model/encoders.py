"""
Encoder classes
"""
import torch
import torch.nn as nn
from parts import DoubleConv, Down

class PastTrajectoryEncoder(nn.Module):
    """
    U-Net encoder for past trajectory
    """
    def __init__(self, in_channels):
        super().__init__()

        # inc = initial conv
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
    
    def forward(self, x):
        f1 = self.inc(x)    # H x W
        f2 = self.down1(f1) # H/2 x W/2
        f3 = self.down2(f2) # H/4 x W/4
        
        # None because we don't have 4 layer and we want to keep the interface consistent
        return f1, f2, f3, None 

class ImpassableEncoder(nn.Module):
    """
    U-Net encoder for impassable objects map
    """
    def __init__(self, in_channels):
        super().__init__()

        # inc = initial conv
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
    
    def forward(self, x):
        f1 = self.inc(x)    # H x W
        f2 = self.down1(f1) # H/2 x W/2
        f3 = self.down2(f2) # H/4 x W/4
        
        # None because we don't have 4 layer and we want to keep the interface consistent
        return f1, f2, f3, None 

class ContextEncoder(nn.Module):
    """
    U-Net encoder for context (full image without detailed zoom)
    """
    def __init__(self, in_channels):
        super().__init__()

        # inc = initial conv
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
    
    def forward(self, x):
        f1 = self.inc(x)    # H x W
        f2 = self.down1(f1) # H/2 x W/2
        f3 = self.down2(f2) # H/4 x W/4
        f4 = self.down3(f3) # H/8 x W/8
        
        return f1, f2, f3, f4

class ZoomEncoder(nn.Module):
    """
    U-Net encoder for zoomed-in image of pedestrian
    """
    def __init__(self, in_channels):
        super().__init__()

        # inc = initial conv
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
    
    def forward(self, x):
        f1 = self.inc(x)    # H x W
        f2 = self.down1(f1) # H/2 x W/2
        f3 = self.down2(f2) # H/4 x W/4
        f4 = self.down3(f3) # H/8 x W/8
        
        return f1, f2, f3, f4