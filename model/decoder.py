"""
Decoder classes
"""
import torch
import torch.nn as nn
from model.parts import Up, OutConv

class Decoder(nn.Module):
    """
    Decoder class for fused features
    """
    def __init__(self, fused_channels):
        super().__init__()
        c1, c2, c3, c4 = fused_channels

        # currently encoders return 4 levels
        self.up1 = Up(c4, c3, c3)
        self.up2 = Up(c3, c2, c2)
        self.up3 = Up(c2, c1, c1)

        self.outc = OutConv(c1, 1)
    
    def forward(self, levels):
        f1, f2, f3, f4 = levels

        x = self.up1(f4, f3)
        x = self.up2(x, f2)
        x = self.up3(x, f1)

        return self.outc(x)