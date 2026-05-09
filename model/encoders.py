"""
Encoder classes.
"""
import torch.nn as nn
from model.parts import DoubleConv, Down


def _w(c, width):
    return max(1, int(c * width))


class PastTrajectoryEncoder(nn.Module):
    """U-Net encoder for past trajectory"""
    def __init__(self, in_channels, width=1.0):
        super().__init__()
        self.inc   = DoubleConv(in_channels,  _w(64,  width))
        self.down1 = Down(_w(64,  width),     _w(128, width))
        self.down2 = Down(_w(128, width),     _w(256, width))

    def forward(self, x):
        f1 = self.inc(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        return f1, f2, f3, None


class ObstacleEncoder(nn.Module):
    """U-Net encoder for obstacle map"""
    def __init__(self, in_channels, width=1.0):
        super().__init__()
        self.inc   = DoubleConv(in_channels,  _w(32,  width))
        self.down1 = Down(_w(32,  width),     _w(64,  width))
        self.down2 = Down(_w(64,  width),     _w(128, width))

    def forward(self, x):
        f1 = self.inc(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        return f1, f2, f3, None


class ContextEncoder(nn.Module):
    """U-Net encoder for full context image"""
    def __init__(self, in_channels, width=1.0):
        super().__init__()
        self.inc   = DoubleConv(in_channels,  _w(64,  width))
        self.down1 = Down(_w(64,  width),     _w(128, width))
        self.down2 = Down(_w(128, width),     _w(256, width))
        self.down3 = Down(_w(256, width),     _w(512, width))

    def forward(self, x):
        f1 = self.inc(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        return f1, f2, f3, f4


class ZoomEncoder(nn.Module):
    """U-Net encoder for zoomed-in pedestrian crop"""
    def __init__(self, in_channels, width=1.0):
        super().__init__()
        self.inc   = DoubleConv(in_channels,  _w(64,  width))
        self.down1 = Down(_w(64,  width),     _w(128, width))
        self.down2 = Down(_w(128, width),     _w(256, width))
        self.down3 = Down(_w(256, width),     _w(512, width))

    def forward(self, x):
        f1 = self.inc(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        return f1, f2, f3, f4