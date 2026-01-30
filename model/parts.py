"""
parts.py
Low-level UNet blocks (Conv, Down, Up...)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Double convolution block: Conv -> Norm -> ReLU (x2)
    """
    def __init__(self, in_ch, out_ch, norm="batch", groups=1):
        super().__init__()

        ## Choose normalization
        if norm == "batch":
            NormLayer = nn.BatchNorm2d
        elif norm == "instance":
            NormLayer = nn.InstanceNorm2d
        else:
            raise ValueError(f"Unknown normalization {norm}")
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, groups=groups),
            NormLayer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=groups),
            NormLayer(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_ch, out_ch, norm="batch", groups=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), # pooling
            DoubleConv(in_ch, out_ch, norm=norm, groups=groups) # conv
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, x_ch, skip_ch, out_ch, bilinear=True, norm="batch"):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(x_ch, x_ch, kernel_size=2, stride=2)

        self.conv = DoubleConv(x_ch + skip_ch, out_ch, norm=norm)

    def forward(self, x, skip): # x: from bottom (decoder), skip: from left
        x = self.up(x)

        # pad x, skip if shape mismatch
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX-diffX // 2,
                    diffY // 2, diffY-diffY // 2])
        
        x = torch.cat([skip, x], dim=1) # skip connection

        return self.conv(x)
        
class OutConv(nn.Module):
    """
    End of architecture, map feature maps to # of classes
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) # pixel-wise prediction

    def forward(self, x):
        return self.conv(x)

# if __name__ == "__main__":
    # x = torch.randn(1, 3, 128, 128) # [B, C, H, W]
    # model = DoubleConv(3, 64, norm="batch") #= torch.Size([1, 64, 128, 128])
    # model = Down(3, 64, norm="batch") #= torch.Size([1, 64, 64, 64])
    # print(model(x).shape)