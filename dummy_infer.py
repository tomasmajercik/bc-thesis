import os
import torch
import numpy as np
import matplotlib.pyplot as plt

##----- Config -----##
NPY_PATH = "data/processed/PETS09/input/0000.npy"
MASK_PATH = "data/processed/PETS09/obstacle_mask.npy"
DEVICE   = "cpu"

##----- Load Data -----##
x = np.load(NPY_PATH)          # (H, W, 7)
H, W, C = x.shape
x2 = np.load(MASK_PATH)     # (H, W, 1)

##----- Split Inputs -----##
local_rgb   = x[:, :, 0:3]     # (H, W, 3)
context_rgb = x[:, :, 3:6]     # (H, W, 3)
traj_raster = x[:, :, 6:7]     # (H, W, 1)
impassable_raster = x2         # (H, W, 1)

##----- To Torch -----##
def to_tensor(x):
    return(
        torch.from_numpy(x)
        .permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        .unsqueeze(0)     # (1, H, W, C) - adds batch dim
        .float().to(DEVICE)
    )

past = to_tensor(traj_raster)
imp  = to_tensor(impassable_raster)
ctx  = to_tensor(context_rgb)
zoom = to_tensor(local_rgb)

##----- Model -----##
from model.model import MultiEncoderUNet

model = MultiEncoderUNet(
    past_channels = 1,
    impassable_channels = 1,
    context_channels = 3,
    zoom_channels = 3
).to(DEVICE); model.eval()

with torch.no_grad():
    out = model(past, imp, ctx, zoom)
print(out.shape)

pred = out.squeeze().cpu().numpy()  # (H, W)
os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(6, 6))
plt.imshow(pred, cmap="hot")
plt.colorbar()
plt.title("Model output (raw)")
plt.tight_layout()

plt.savefig("outputs/prediction.png", dpi=150)
plt.close()