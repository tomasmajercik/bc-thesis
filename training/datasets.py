"""datasets.py"""
import os
import torch
import numpy as np
import torch.nn.functional as F

class PETSDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="data/processed/PETS09", scale=0.5, return_coords=False):
        self.scale = scale
        self.return_coords = return_coords

        self.input_dir = os.path.join(root_dir, "input")
        self.target_dir = os.path.join(root_dir, "target")
        self.coords_dir = os.path.join(root_dir, "target_coords")

        self.files = sorted([
            f for f in os.listdir(self.input_dir)
            if f.endswith(".npy")
        ])

        mask = np.load(os.path.join(root_dir, "obstacle_mask.npy"))
        mask = mask.squeeze(-1)

        # map colors to class ids
        mask_class = np.zeros_like(mask, dtype=np.float32)
        mask_class[mask == 0] = 0.0        # obstacle
        mask_class[mask == 128] = 0.5      # grass (walkable but not main path)
        mask_class[mask == 255] = 1.0      # road

        self.mask = torch.from_numpy(mask_class).unsqueeze(0)

    def _resize(self, t, scale):
        return F.interpolate(
            t.unsqueeze(0),
            scale_factor=scale,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x_path = os.path.join(self.input_dir, self.files[idx])
        y_path = os.path.join(self.target_dir, self.files[idx])

        x = np.load(x_path)
        y = np.load(y_path)

        target = torch.from_numpy(y).unsqueeze(0).float() /255.0
        zoom   = torch.from_numpy(x[:, :, 0:3]).permute(2,0,1).float() /255.0
        ctx    = torch.from_numpy(x[:, :, 3:6]).permute(2,0,1).float() /255.0
        past   = torch.from_numpy(x[:, :, 6:7]).permute(2,0,1).float() /255.0
        impass = self.mask

        if self.scale != 1:
            past   = self._resize(past, self.scale)
            impass = self._resize(impass, self.scale)
            ctx    = self._resize(ctx, self.scale)
            zoom   = self._resize(zoom, self.scale)
            target = self._resize(target, self.scale)

        if self.return_coords:
            c_path = os.path.join(self.coords_dir, self.files[idx])
            c = np.load(c_path)
            coords = torch.from_numpy(c).float()
            if self.scale != 1:
                coords = coords * self.scale

            # Pad to fixed length so DataLoader can collate
            max_steps = 15  # set to your future_steps config value
            n = coords.shape[0]
            if n < max_steps:
                pad = torch.full((max_steps - n, 2), -1.0)  # -1 flags invalid/missing
                coords = torch.cat([coords, pad], dim=0)

            return past, impass, ctx, zoom, target, coords    

        return past, impass, ctx, zoom, target
            
    
if __name__ == "__main__":
    ds = PETSDataset()
    PETSDataset.__getitem__(ds, 0)