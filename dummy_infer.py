import numpy as np
import torch
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
NPY_PATH = "data/processed/PETS09/input/0000.npy"
DEVICE   = "cpu"   # alebo "cuda"

# ----------------------------
# LOAD DATA
# ----------------------------
x = np.load(NPY_PATH)          # (H, W, 7)
H, W, C = x.shape
assert C == 7, f"Expected 7 channels, got {C}"

# split channels
local_rgb   = x[:, :, 0:3]     # (H, W, 3)
context_rgb = x[:, :, 3:6]     # (H, W, 3)
traj_raster = x[:, :, 6:7]     # (H, W, 1)

impassable_raster_path = "data/processed/PETS09/obstacle_mask.npy"
impassable_raster = np.load(impassable_raster_path)  # (H, W, 1 )

# ----------------------------
# TO TORCH
# ----------------------------
def to_tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

local_t   = to_tensor(local_rgb).unsqueeze(0)     # (1, 3, H, W)
context_t = to_tensor(context_rgb).unsqueeze(0)   # (1, 3, H, W)
traj_t    = to_tensor(traj_raster).unsqueeze(0)   # (1, 1, H, W)
impassable_t = to_tensor(impassable_raster).unsqueeze(0)  # (1, 1, H, W)

# ----------------------------
# MODEL
# ----------------------------
from model.model import MultiEncoderUNet

model = MultiEncoderUNet(
    past_channels=1,
    context_channels=3,
    zoom_channels=3,
    impassable_channels=1,
    fusion_type="attention"
).to(DEVICE)

model.eval()

# ----------------------------
# INFERENCE
# ----------------------------
with torch.no_grad():
    out = model(
        past=traj_t.to(DEVICE),
        context=context_t.to(DEVICE),
        zoom=local_t.to(DEVICE),
        impassable=impassable_t.to(DEVICE)
    )

out = out.squeeze().cpu().numpy()  # (H, W)

print("Output:", out.shape, out.min(), out.max())

# ----------------------------
# VISUALIZATION
# ----------------------------
fig, axs = plt.subplots(1, 4, figsize=(16, 4))

axs[0].set_title("Local RGB")
axs[0].imshow(local_rgb)
axs[0].axis("off")

axs[1].set_title("Context RGB")
axs[1].imshow(context_rgb)
axs[1].axis("off")

axs[2].set_title("Past trajectory")
axs[2].imshow(traj_raster.squeeze(), cmap="gray")
axs[2].axis("off")

axs[3].set_title("Model output (raw)")
axs[3].imshow(out, cmap="hot")
axs[3].axis("off")

plt.tight_layout()
plt.show()
