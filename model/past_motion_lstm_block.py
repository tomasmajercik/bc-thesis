""" Past motion LSTM block for processing past trajectory information.
It was only tested, but is not used in the final model.
"""

import torch
import torch.nn as nn

class PastMotionBlock(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_size, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, coords, img_h, img_w):
        B, T, _ = coords.shape

        norm = torch.tensor([img_w, img_h], dtype=torch.float32, device=coords.device)
        pos  = coords / norm # (B, T, 2)

        vel = torch.zeros_like(pos)
        vel[:, 1:] = pos[:, 1:] - pos[:, :-1]

        inp = torch.cat([pos, vel], dim=-1) # (B, T, 4): [x, y, dx, dy]
        
        _, (h_n, _) = self.lstm(inp) # h_n: (1, B, hidden_size)
        h = h_n.squeeze(0) # (B, hidden_size)
        h = self.mlp(h) # (B, hidden_size)
        
        return h