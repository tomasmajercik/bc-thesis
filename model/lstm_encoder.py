"""
lstm_encoder.py  LSTM-based past trajectory encoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _w(c, width):
    return max(1, int(c * width))


class LSTMTrajectoryEncoder(nn.Module):
    """
    LSTM encoder for raw past trajectory coordinates.

    Input : coords (B, T, 2)  — (x, y) positions in pixel space
    Output: (f1, f2, f3, None) with shapes
            f1  (B, 64*width,  H,    W   )
            f2  (B, 128*width, H//2, W//2)
            f3  (B, 256*width, H//4, W//4)
    where H and W are provided at forward time.
    """

    def __init__(self, past_traj_steps, hidden_size=256, width=1.0):
        super().__init__()
        self.hidden_size = hidden_size

        # LSTM over 4-dim input: (x_norm, y_norm, dx_norm, dy_norm)
        self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_size, batch_first=True)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )

        # Per-scale linear projections: hidden → channel count for that scale
        self.proj1 = nn.Linear(hidden_size, _w(64,  width))
        self.proj2 = nn.Linear(hidden_size, _w(128, width))
        self.proj3 = nn.Linear(hidden_size, _w(256, width))

    def forward(self, coords, img_h, img_w):
        """
        coords : (B, T, 2)  pixel-space (x, y)
        img_h  : int  — full-resolution height H
        img_w  : int  — full-resolution width  W
        Returns (f1, f2, f3, None) matching PastTrajectoryEncoder's interface.
        """
        B, T, _ = coords.shape

        # Finite-difference velocities; first timestep has zero velocity
        vel = torch.zeros_like(coords)
        vel[:, 1:] = coords[:, 1:] - coords[:, :-1]

        # Normalize position and velocity to [0, 1]
        x_n  = coords[..., 0:1] / img_w
        y_n  = coords[..., 1:2] / img_h
        dx_n = vel[..., 0:1]    / img_w
        dy_n = vel[..., 1:2]    / img_h

        inp = torch.cat([x_n, y_n, dx_n, dy_n], dim=-1)  # (B, T, 4)

        _, (h_n, _) = self.lstm(inp)   # h_n: (1, B, hidden_size)
        h = h_n.squeeze(0)             # (B, hidden_size)

        # MLP → per-scale projection → reshape (B, C, 1, 1) → upsample
        h_feat = self.mlp(h)

        f1 = F.interpolate(
            self.proj1(h_feat).view(B, -1, 1, 1),
            size=(img_h,       img_w      ),
            mode="bilinear", align_corners=False,
        )
        f2 = F.interpolate(
            self.proj2(h_feat).view(B, -1, 1, 1),
            size=(img_h // 2,  img_w // 2 ),
            mode="bilinear", align_corners=False,
        )
        f3 = F.interpolate(
            self.proj3(h_feat).view(B, -1, 1, 1),
            size=(img_h // 4,  img_w // 4 ),
            mode="bilinear", align_corners=False,
        )

        return f1, f2, f3, None
