import os
import torch
import numpy as np
from utils import np_2_tensor, ConsoleColors

class PETSDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="../data/processed/PETS09"):
        self.input_dir = os.path.join(root_dir, "input")
        self.target_dir = os.path.join(root_dir, "target")

        self.files = sorted([
            f for f in os.listdir(self.input_dir)
            if f.endswith(".npy")
        ])

        self.mask = torch.from_numpy(
            np.load(os.path.join(root_dir, "obstacle_mask.npy"))
        ).permute(2,0,1).float()

        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x_path = os.path.join(self.input_dir, self.files[idx])
        y_path = os.path.join(self.target_dir, self.files[idx])

        x = np.load(x_path)
        y = np.load(y_path)


        zoom   = torch.from_numpy(x[:, :, 0:3]).permute(2,0,1).float()
        ctx    = torch.from_numpy(x[:, :, 3:6]).permute(2,0,1).float()
        past   = torch.from_numpy(x[:, :, 6:7]).permute(2,0,1).float()
        impass = self.mask
        target = torch.from_numpy(y).unsqueeze(0).float()

        return past, impass, ctx, zoom, target
    
if __name__ == "__main__":
    ds = PETSDataset()
    PETSDataset.__getitem__(ds, 0)