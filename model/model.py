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
        use_lstm=False,     # True: adds LSTMTrajectoryEncoder as a 5th encoder alongside past_enc
        past_traj_steps=21, # used only when use_lstm=True
        lstm_hidden=256,    # used only when use_lstm=True
    ):
        super().__init__()
        self.use_lstm = use_lstm

        # ---------- Encoders ----------
        self.past_enc   = PastTrajectoryEncoder(past_channels,     width=width)
        self.impass_enc = ObstacleEncoder(obstacle_channels,       width=width)
        self.ctx_enc    = ContextEncoder(context_channels,         width=width)
        self.zoom_enc   = ZoomEncoder(zoom_channels,               width=width)

        if use_lstm:
            self.lstm_enc = LSTMTrajectoryEncoder(
                past_traj_steps=past_traj_steps,
                hidden_size=lstm_hidden,
                width=width,
            )

        # ---------- Fusion ----------
        # f4 only has ctx + zoom (past, impass, lstm encoders return None at level 4)
        if use_lstm:
            # 5 encoders: [past, impass, ctx, zoom, lstm]
            self.fusion = AttentionFusion([
                [_w(64,  width), _w(32,  width), _w(64,  width), _w(64,  width), _w(64,  width)],   # f1
                [_w(128, width), _w(64,  width), _w(128, width), _w(128, width), _w(128, width)],   # f2
                [_w(256, width), _w(128, width), _w(256, width), _w(256, width), _w(256, width)],   # f3
                [_w(512, width), _w(512, width)],                                                     # f4
            ])
            fused_channels = [
                _w(64,  width) + _w(32,  width) + _w(64,  width) + _w(64,  width) + _w(64,  width),  # f1: 288 @ base
                _w(128, width) + _w(64,  width) + _w(128, width) + _w(128, width) + _w(128, width),  # f2: 576 @ base
                _w(256, width) + _w(128, width) + _w(256, width) + _w(256, width) + _w(256, width),  # f3: 1152 @ base
                _w(512, width) + _w(512, width),                                                       # f4: 1024 @ base
            ]
        else:
            # 4 encoders: [past, impass, ctx, zoom]
            self.fusion = AttentionFusion([
                [_w(64,  width), _w(32,  width), _w(64,  width), _w(64,  width)],  # f1
                [_w(128, width), _w(64,  width), _w(128, width), _w(128, width)],  # f2
                [_w(256, width), _w(128, width), _w(256, width), _w(256, width)],  # f3
                [_w(512, width), _w(512, width)],                                   # f4
            ])
            fused_channels = [
                _w(64,  width) + _w(32,  width) + _w(64,  width) + _w(64,  width),  # f1: 224 @ base
                _w(128, width) + _w(64,  width) + _w(128, width) + _w(128, width),  # f2: 448 @ base
                _w(256, width) + _w(128, width) + _w(256, width) + _w(256, width),  # f3: 896 @ base
                _w(512, width) + _w(512, width),                                      # f4: 1024 @ base
            ]

        # ---------- Decoder ----------
        self.decoder = Decoder(fused_channels)

    def forward(self, past, imp, ctx, zoom, past_coords=None, return_attention=False):
        e1 = self.past_enc(past)
        e2 = self.impass_enc(imp)
        e3 = self.ctx_enc(ctx)
        e4 = self.zoom_enc(zoom)

        if self.use_lstm and past_coords is not None:
            H, W = imp.shape[2], imp.shape[3]
            e5 = self.lstm_enc(past_coords, H, W)
            fused_feats, attention_weights = self.fusion([e1, e2, e3, e4, e5])
        else:
            fused_feats, attention_weights = self.fusion([e1, e2, e3, e4])

        out = self.decoder(fused_feats)

        if return_attention:
            return out, attention_weights
        return out


# ---------- quick sanity check ----------
if __name__ == "__main__":
    B, H, W, T = 1, 256, 256, 14
    past        = torch.randn(B, 1, H, W)
    imp         = torch.randn(B, 1, H, W)
    ctx         = torch.randn(B, 3, H, W)
    zoom        = torch.randn(B, 3, H, W)
    past_coords = torch.rand(B, T, 2) * torch.tensor([W, H], dtype=torch.float32)

    print("=== use_lstm=False ===")
    for label, width in [("small", 0.5), ("base", 1.0), ("large", 2.0)]:
        model  = MultiEncoderUNet(width=width, use_lstm=False)
        out    = model(past, imp, ctx, zoom)
        params = sum(p.numel() for p in model.parameters())
        print(f"[{label:>5}]  width={width}  output={tuple(out.shape)}  params={params/1e6:.2f}M")

    print("=== use_lstm=True (5th encoder) ===")
    for label, width in [("small", 0.5), ("base", 1.0), ("large", 2.0)]:
        model  = MultiEncoderUNet(width=width, use_lstm=True, past_traj_steps=T)
        out    = model(past, imp, ctx, zoom, past_coords)
        params = sum(p.numel() for p in model.parameters())
        print(f"[{label:>5}]  width={width}  output={tuple(out.shape)}  params={params/1e6:.2f}M")
