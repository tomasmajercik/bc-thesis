"""
Fusion and Attention mechanisms
"""
import torch
import torch.nn as nn

class EncoderAttention(nn.Module):
    """
    Encoder-wise attention for ONE feature level.
    Learns how mmuch each encoder contributes
    """
    def __init__(self, in_channels_list):
        super().__init__()
        self.total_channels = sum(in_channels_list)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.total_channels, len(in_channels_list))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features):
        """
        features: list of tensors [(B, C_i, H, W), ...]
        """
        
        pooled = [self.pool(f).flatten(1) for f in features] # (B, C_i)
        x = torch.cat(pooled, dim=1) # (B, sum(C_i))
        weights = self.softmax(self.fc(x)) # (B, N)

        return weights

class AttentionFusion(nn.Module):
    """
    Applies encoted-wise attention at each feature scale and 
    contatenates weighted features
    """
    def __init__(self, fused_channels):
        """
        fused_channels: list like [C1, C2, C3, C4]
        (sum of encoder channels per level)
        """
        super().__init__()

        self.attentions = nn.ModuleList()
        for level_channels in fused_channels:
            self.attentions.append(
                EncoderAttention(level_channels)
            )

    def forward(self, enc_outputs):
        """
        enc_outputs:
            list of encoders
            each encoder -> (f1, f2, f3, f4)

        Returns:
            fused_feats: (f1, f2, f3, f4)
        """
        num_levels = len(enc_outputs[0])
        fused_feats = []

        for level in range(num_levels):
            level_feats = [
                enc[level] for enc in enc_outputs
                if enc[level] is not None
            ]

            att = self.attentions[level](level_feats) # (B, N)

            weighted = []
            for i, f in enumerate(level_feats):
                w = att[:, i].view(-1, 1, 1, 1)
                weighted.append(f*w)

            fused = torch.cat(weighted, dim=1)
            fused_feats.append(fused)
    
        return fused_feats, None