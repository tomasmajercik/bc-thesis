import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, hidden_size, encoder_channels):
        super().__init__()

        self.projections = nn.ModuleList()
        for level in encoder_channels:          # interaters through levels (f1, f2, f3)
            level_projs = nn.ModuleList()
            for C in level:                     # iterate encoders within level
                level_projs.append(nn.Linear(hidden_size, 2 * C)) # 2*c to produce both, gamma and beta
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


        