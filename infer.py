import time
import torch
from torch.utils.data import DataLoader
from model.model import MultiEncoderUNet
from training.datasets import PETSDataset

##----- Config -----##
INPUT_PATH = "data/processed/PETS09/input/"
MASK_PATH = "data/processed/PETS09/obstacle_mask.npy"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

##----- Load Data -----##
dataset = PETSDataset(scale=1)
samples = DataLoader(dataset, batch_size=1, shuffle=False)

## Load model
model = MultiEncoderUNet(
    past_channels = 1,
    impassable_channels = 1,
    context_channels = 3,
    zoom_channels = 3
).to(DEVICE)

# Load checkpoint
ckpt_path = "checkpoints/long-strict-w-imgs/best_model.pth"
checkpoint = torch.load(ckpt_path, map_location=DEVICE)

# Common checkpoint formats
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

no_samples = 20

start_time = time.time()
for i, sample in enumerate(samples):
    past, imp, ctx, zoom, target = [x.to(DEVICE) for x in sample]

    with torch.no_grad():
        pred = model(past, imp, ctx, zoom)

    if i > no_samples-1:
        break
end_time = time.time()
    
print()
print("-"*20, f"on {DEVICE}", "-"*20)
print(f"Samples per second:                {(no_samples / (end_time - start_time)):.2f} samples/s")
print(f"Inference time for {no_samples} samples:     {end_time - start_time:.2f} s")
print(f"Average inference time per sample: {(end_time - start_time) / no_samples:.2f} s/sample")
print("-"*50)
print()



   