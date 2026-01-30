"""
Fusion and Attention mechanisms
"""
import torch
import torch.nn as nn

class SimpleConcatFusion(nn.Module):
    """
    Concatenates available encoder features at each scale
    """
    def forward(self, encoder_outputs):
        # encoder_outputs: list of (f1, f2, f3, f4)
        fused = []
        fused_channels = []
        num_levels = len(encoder_outputs[0])  # 4

        for level in range(num_levels):
            feats = [enc[level] for enc in encoder_outputs if enc[level] is not None]
            
            fused_feat = torch.cat(feats, dim=1)
            fused.append(fused_feat)
            fused_channels.append(fused_feat.shape[1])
        
        return fused, fused_channels


class EncoderRelevanceAttention(nn.Module):
    """
    Computes attention weights over multiple encoder feature maps
    (encoder-level / modality-level attention)
    """
    def __init__(self, in_channels_list):
        super().__init__()
        self.num_encoders = len(in_channels_list)

        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.linear   = nn.Linear(sum(in_channels_list), self.num_encoders)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features):
        """
        features: list of tensors [B, C_i, H, W]
        """
        pooled = []
        for f in features:
            x = self.pool(f)    # [B, C, 1, 1]
            x = x.squeeze(-1).squeeze(-1) # [B, C]
            pooled.append(x)

        x = torch.cat(pooled, dim=1) # [B, sum(C)]
        weights = self.softmax(self.linear(x)) # [B, N]

        return weights

class AttentionFusion(nn.Module):
    """
    Attention-based fusion of encoder features (per level)
    """
    def __init__(self, in_channels_list):
        super().__init__()
        self.att = EncoderRelevanceAttention(in_channels_list)

    def forward(self, encoder_outputs):
        """
        encoder_outputs: list of (f1, f2, f3, f4)
        """
        num_levels = len(encoder_outputs[0])
        fused = []
        fused_channels = []

        for level in range(num_levels):
            feats = [enc[level] for enc in encoder_outputs if enc[level] is not None]

            # attention weights [B, N]
            weights = self.att(feats)

            # weighted sum
            fused_feat = 0
            for i, f in enumerate(feats):
                w = weights[:, i].view(-1, 1, 1, 1)
                fused_feat = fused_feat + w * f

            fused.append(fused_feat)
            fused_channels.append(fused_feat.shape[1])

        return fused, fused_channels