"""
model.py  High-level model module.
"""
import torch
import torch.nn as nn
from model.encoders import (
    PastTrajectoryEncoder,
    ObstacleEncoder,
    ContextEncoder,
    ZoomEncoder,
)
from model.lstm_encoder import LSTMTrajectoryEncoder
from model.atention import AttentionFusion
from model.decoder import Decoder


def _w(c, width):
    return max(1, int(c * width))


class MultiEncoderUNet(nn.Module):
    def __init__(
        self,
        past_channels=1,
        obstacle_channels=1,
        context_channels=3,
        zoom_channels=3,
        width=1.0,          # 0.5 = small  |  1.0 = base  |  2.0 = large
        use_lstm=False,     # True: past_enc is LSTMTrajectoryEncoder
        past_traj_steps=21, # used only when use_lstm=True
        lstm_hidden=256,    # used only when use_lstm=True
    ):
        super().__init__()
        self.use_lstm = use_lstm

        # ---------- Encoders ----------
        if use_lstm:
            self.past_enc = LSTMTrajectoryEncoder(
                past_traj_steps=past_traj_steps,
                hidden_size=lstm_hidden,
                width=width,
            )
        else:
            self.past_enc = PastTrajectoryEncoder(past_channels, width=width)
        self.impass_enc = ObstacleEncoder(obstacle_channels,       width=width)
        self.ctx_enc    = ContextEncoder(context_channels,         width=width)
        self.zoom_enc   = ZoomEncoder(zoom_channels,               width=width)

        # ---------- Fusion ----------
        # Channel lists per level — order matches forward(): [past, impass, ctx, zoom]
        # f4 only has ctx + zoom (past and impass encoders return None at level 4)
        self.fusion = AttentionFusion([
            [_w(64,  width), _w(32,  width), _w(64,  width), _w(64,  width)],  # f1
            [_w(128, width), _w(64,  width), _w(128, width), _w(128, width)],  # f2
            [_w(256, width), _w(128, width), _w(256, width), _w(256, width)],  # f3
            [_w(512, width), _w(512, width)],                                   # f4
        ])

        # ---------- Decoder ----------
        fused_channels = [
            _w(64,  width) + _w(32,  width) + _w(64,  width) + _w(64,  width),  # f1: 224 @ base
            _w(128, width) + _w(64,  width) + _w(128, width) + _w(128, width),  # f2: 448 @ base
            _w(256, width) + _w(128, width) + _w(256, width) + _w(256, width),  # f3: 896 @ base
            _w(512, width) + _w(512, width),                                      # f4: 1024 @ base
        ]
        self.decoder = Decoder(fused_channels)

    def forward(self, past, imp, ctx, zoom, return_attention=False):
        if self.use_lstm:
            H, W = imp.shape[2], imp.shape[3]
            e1 = self.past_enc(past, H, W)
        else:
            e1 = self.past_enc(past)
        e2 = self.impass_enc(imp)
        e3 = self.ctx_enc(ctx)
        e4 = self.zoom_enc(zoom)

        fused_feats, attention_weights = self.fusion([e1, e2, e3, e4])

        out = self.decoder(fused_feats)

        if return_attention:
            return out, attention_weights
        return out


# ---------- quick sanity check ----------
if __name__ == "__main__":
    B, H, W, T = 1, 256, 256, 20
    past  = torch.randn(B, 1, H, W)
    imp   = torch.randn(B, 1, H, W)
    ctx   = torch.randn(B, 3, H, W)
    zoom  = torch.randn(B, 3, H, W)
    coords = torch.rand(B, T, 2) * torch.tensor([W, H], dtype=torch.float32)

    print("=== use_lstm=False ===")
    for label, width in [("small", 0.5), ("base", 1.0), ("large", 2.0)]:
        model  = MultiEncoderUNet(width=width, use_lstm=False)
        out    = model(past, imp, ctx, zoom)
        params = sum(p.numel() for p in model.parameters())
        print(f"[{label:>5}]  width={width}  output={tuple(out.shape)}  params={params/1e6:.2f}M")

    print("=== use_lstm=True ===")
    for label, width in [("small", 0.5), ("base", 1.0), ("large", 2.0)]:
        model  = MultiEncoderUNet(width=width, use_lstm=True, past_traj_steps=T)
        out    = model(coords, imp, ctx, zoom)
        params = sum(p.numel() for p in model.parameters())
        print(f"[{label:>5}]  width={width}  output={tuple(out.shape)}  params={params/1e6:.2f}M")