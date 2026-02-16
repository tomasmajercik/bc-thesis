"""datasets.py"""
import os
import torch
import numpy as np
import torch.nn.functional as F

class PETSDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="data/processed/PETS09", scale=1):
        self.scale = scale
        self.input_dir = os.path.join(root_dir, "input")
        self.target_dir = os.path.join(root_dir, "target")

        self.files = sorted([
            f for f in os.listdir(self.input_dir)
            if f.endswith(".npy")
        ])

        self.mask = torch.from_numpy(
            np.load(os.path.join(root_dir, "obstacle_mask.npy"))
        ).permute(2,0,1).float()

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

        target = torch.from_numpy(y).unsqueeze(0).float() / 255.0
        if target.sum() < 1e-6:  # essentially all zeros
            return self.__getitem__((idx + 1) % len(self.files))  # return next sample

        zoom   = torch.from_numpy(x[:, :, 0:3]).permute(2,0,1).float()
        ctx    = torch.from_numpy(x[:, :, 3:6]).permute(2,0,1).float()
        past   = torch.from_numpy(x[:, :, 6:7]).permute(2,0,1).float()
        impass = self.mask

        if self.scale != 1:
            past   = self._resize(past, self.scale)
            impass = self._resize(impass, self.scale)
            ctx    = self._resize(ctx, self.scale)
            zoom   = self._resize(zoom, self.scale)
            target = self._resize(target, self.scale)

        return past, impass, ctx, zoom, target
    
if __name__ == "__main__":
    ds = PETSDataset()
    PETSDataset.__getitem__(ds, 0)