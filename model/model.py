"""
model.py High-level model module.
"""
import torch
import torch.nn as nn
from model.encoders import (
    PastTrajectoryEncoder,
    ObstacleEncoder,
    ContextEncoder,
    ZoomEncoder
)
from model.atention import AttentionFusion
from model.decoder import Decoder

class MultiEncoderUNet(nn.Module):
    def __init__(
        self, 
        past_channels=1,
        obstacle_channels=1,
        context_channels=3,
        zoom_channels=3
    ):
        super().__init__()

        ## ---------- Encoders (down) ---------- ##
        self.past_enc   = PastTrajectoryEncoder(in_channels=past_channels) 
        self.impass_enc = ObstacleEncoder(in_channels=obstacle_channels)
        self.ctx_enc    = ContextEncoder(in_channels=context_channels)
        self.zoom_enc   = ZoomEncoder(in_channels=zoom_channels)

        # ---- Fusion + Decoder ----
        self.fusion = AttentionFusion([
            [64, 64, 32, 64],        # f1 
            [128, 128, 64, 128],     # f2
            [256, 256, 128, 256],    # f3
            [512, 512]               # f4 (only ctx + zoom)
        ])

        ## ---------- Decoder (up&out) ---------- ##
        self.fused_channels = [224, 448, 896, 1024] # No. channels in such level (sum of output channels)
        self.decoder = Decoder(self.fused_channels)

    def forward(self, past, imp, ctx, zoom, return_attention=False):
        # Encode
        e1 = self.past_enc(past)
        e2 = self.impass_enc(imp)
        e3 = self.ctx_enc(ctx)
        e4 = self.zoom_enc(zoom)

        # Fuse
        fused_feats, attention_weights = self.fusion([e1, e2, e3, e4])

        # Decode
        out = self.decoder(fused_feats)
        if return_attention:
            return out, attention_weights

        return out

# dummy example
if __name__ == "__main__":
    model = MultiEncoderUNet(
        past_channels = 1,
        obstacle_channels = 1,
        context_channels = 3,
        zoom_channels = 3
    )

    B, H, W = 1, 256, 256

    past = torch.randn(B, 1, H, W)
    imp  = torch.randn(B, 1, H, W)
    ctx  = torch.randn(B, 3, H, W)
    zoom = torch.randn(B, 3, H, W)

    out = model(past, imp, ctx, zoom)

    print("Output shape:", out.shape)


# NO of params
# num_params = sum(p.numel() for p in model.parameters())
# print(f"Params: {num_params/1e6:.2f}M")