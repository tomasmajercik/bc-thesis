""" Feature-wIse Linear Modulation (FiLM) implementation.
It was only tested, but is not used in the final model.
"""
import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, hidden_size, encoder_channels):
        super().__init__()

        self.projections = nn.ModuleList()
        for level in encoder_channels:          # interaters through levels (f1, f2, f3)
            level_projs = nn.ModuleList()
            for C in level:
                proj = nn.Linear(hidden_size, 2 * C)
                nn.init.zeros_(proj.weight)
                nn.init.constant_(proj.bias, 0.0)
                proj.bias.data[:C] = 1.0   # gamma starts at 1 (identity)
                # proj.bias.data[C:] stays 0  # beta starts at 0 (no shift)
                level_projs.append(proj)
            self.projections.append(level_projs)

    def forward(self, h, encoder_features):
        """
        h: (B, hidden_size)
        encoder_features: list of levels, each level is list of feature maps
        [[f1_past, f1_impass, f1_ctx, f1_zoom],
         [f2_past, f2_impass, f2_ctx, f2_zoom],
         [f3_past, f3_impass, f3_ctx, f3_zoom]]
        """
        
        modulated = []
        for level_idx, level_feats in enumerate(encoder_features):
            level_modulated = []
            for enc_idx, feat in enumerate(level_feats):
                out = self.projections[level_idx][enc_idx](h) # (B, 2*C) # type: ignore 
                
                gamma, beta = out.chunk(2, dim=-1)            # each (B, C)
                gamma = gamma.unsqueeze(-1).unsqueeze(-1)     # (B, C, 1, 1)
                beta  = beta.unsqueeze(-1).unsqueeze(-1)      # (B, C, 1, 1)

                level_modulated.append(gamma * feat + beta)
            modulated.append(level_modulated)
        
        return modulated

if __name__ == "__main__":
    import torch
    film = FiLM(hidden_size=256, encoder_channels=[[64, 32], [128, 64]])
    h = torch.randn(2, 256)  # random h
    feat = torch.randn(2, 64, 32, 32)  # random feature map
    features = [[feat, torch.randn(2, 32, 32, 32)], 
                [torch.randn(2, 128, 16, 16), torch.randn(2, 64, 16, 16)]]
    out = film(h, features)
    diff = (out[0][0] - feat).abs().max()
    print(f"Max diff from identity at init: {diff.item()}")  # should be ~0.0

        