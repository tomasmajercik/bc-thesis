"""
High-level model module.
"""
import torch
import torch.nn as nn
from encoders import (
    PastTrajectoryEncoder,
    ImpassableEncoder,
    ContextEncoder,
    ZoomEncoder
)
from atention import AttentionFusion
from decoder import Decoder

class MultiEncoderUNet(nn.Module):
    def __init__(
        self, 
        past_channels,
        impassable_channels,
        context_channels,
        zoom_channels,
        fusion_type="attention" # or "concat" (not imported)
    ):
        super().__init__()

        ## ---------- Encoders (down) ---------- ##
        self.past_enc   = PastTrajectoryEncoder(past_channels) 
        self.impass_enc = ImpassableEncoder(impassable_channels)
        self.ctx_enc    = ContextEncoder(context_channels)
        self.zoom_enc   = ZoomEncoder(zoom_channels)

        ## ---------- Fusion (neck) ---------- ##
        in_channels = [
            [64, 128, 256, None]    # past
            [32, 64, 128, None]     # context
            [64, 128, 256, 512]     # impassable
            [64, 128, 256, 512]     # zoom
        ]

        # This block expects fixed list of channels
        fused_channels = [
            sum(ch for ch in level if ch is not None)
            for level in zip(*in_channels) # (tuple of channels)
        ]

        self.fusion = AttentionFusion(fused_channels)

        ## ---------- Decoder (up&out) ---------- ##
        self.decoder = Decoder(fused_channels)

    def forward(self, past, impassable, context, zoom):
        enc_outputs = []

        enc_outputs.append(self.past_enc(past))
        enc_outputs.append(self.impass_enc(impassable))
        enc_outputs.append(self.ctx_enc(context))
        enc_outputs.append(self.zoom_enc(zoom))

        fused_feats, fused_channels = self.fusion(enc_outputs)

        logits = self.decoder(fused_feats)

        return logits





