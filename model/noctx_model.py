"""
model.py  High-level model module.
"""
import torch
import torch.nn as nn
from model.encoders import (
    PastTrajectoryEncoder,
    ObstacleEncoder,
    ZoomEncoder,
)
from model.film import FiLM
from model.decoder import Decoder
from model.atention import AttentionFusion
from model.past_motion_lstm_block import PastMotionBlock

def _w(c, width):
    return max(1, int(c * width))


class MultiEncoderUNet(nn.Module):
    def __init__(
        self,
        past_channels=1,
        obstacle_channels=1,
        zoom_channels=3,
        width=1.0,          # 0.5 = small  |  1.0 = base  |  2.0 = large
        use_motion=False,
        lstm_hidden=256,
    ):
        super().__init__()
        self.use_motion = use_motion

        # ---------- Encoders ----------
        self.past_enc   = PastTrajectoryEncoder(past_channels,     width=width)
        self.impass_enc = ObstacleEncoder(obstacle_channels,       width=width)
        self.zoom_enc   = ZoomEncoder(zoom_channels,               width=width)

        if use_motion:
            self.motion_block = PastMotionBlock(hidden_size=lstm_hidden)
            self.film = FiLM(
                hidden_size=lstm_hidden,
                encoder_channels=[
                    [_w(64,width), _w(32,width), _w(64,width)],   # f1
                    [_w(128,width),_w(64,width), _w(128,width)],  # f2
                    [_w(256,width),_w(128,width),_w(256,width)],  # f3
                ]
            )

        # ---------- Fusion ----------
        # 3 encoders: [past, impass, zoom]
        self.fusion = AttentionFusion([
            [_w(64,  width), _w(32,  width), _w(64,  width)],  # f1
            [_w(128, width), _w(64,  width), _w(128, width)],  # f2
            [_w(256, width), _w(128, width), _w(256, width)],  # f3
            [_w(512, width)],                                   # f4: zoom only
        ])
        fused_channels = [
            _w(64,  width) + _w(32,  width) + _w(64,  width),  # f1: 160 @ base
            _w(128, width) + _w(64,  width) + _w(128, width),  # f2: 320 @ base
            _w(256, width) + _w(128, width) + _w(256, width),  # f3: 640 @ base
            _w(512, width),                                     # f4: 512 @ base
        ]

        # ---------- Decoder ----------
        self.decoder = Decoder(fused_channels)

    def forward(self, past, imp, zoom, past_coords=None, return_attention=False):
        e1 = self.past_enc(past)
        e2 = self.impass_enc(imp)
        e3 = self.zoom_enc(zoom)

        if self.use_motion and past_coords is not None:
            H, W = past.shape[2], past.shape[3]
            h = self.motion_block(past_coords, H, W) # (B, hidden)

            # regroup by level for FiLM modulation (skip f4 None)
            encoder_features = [
                [e1[0], e2[0], e3[0]], # f1
                [e1[1], e2[1], e3[1]], # f2
                [e1[2], e2[2], e3[2]], # f3
            ]
            modulated = self.film(h, encoder_features)

            # replace original features with modulated ones
            e1 = (modulated[0][0], modulated[1][0], modulated[2][0], None)
            e2 = (modulated[0][1], modulated[1][1], modulated[2][1], None)
            e3 = (modulated[0][2], modulated[1][2], modulated[2][2], e3[3])

        fused_feats, attention_weights = self.fusion([e1, e2, e3])

        out = self.decoder(fused_feats)

        if return_attention:
            return out, attention_weights
        return out


# ---------- quick sanity check ----------
if __name__ == "__main__":
    B, H, W, T = 1, 256, 256, 14
    past        = torch.randn(B, 1, H, W)
    imp         = torch.randn(B, 1, H, W)
    zoom        = torch.randn(B, 3, H, W)
    past_coords = torch.rand(B, T, 2) * torch.tensor([W, H], dtype=torch.float32)

    print("=== use_motion=False ===")
    for label, width in [("small", 0.5), ("base", 1.0), ("large", 2.0)]:
        model  = MultiEncoderUNet(width=width, use_motion=False)
        out    = model(past, imp, zoom)
        params = sum(p.numel() for p in model.parameters())
        print(f"[{label:>5}]  width={width}  output={tuple(out.shape)}  params={params/1e6:.2f}M")