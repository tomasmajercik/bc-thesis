import time
import torch
from torch.utils.data import DataLoader

from model.model import MultiEncoderUNet
from training.datasets import PETSDataset


## ----- Config ----- ##
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NO_SAMPLES = 20


## ----- Load Data ----- ##
dataset = PETSDataset(scale=1)

samples = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,          
    pin_memory=True,        
    persistent_workers=True 
)


## ----- Load Model ----- ##
model = MultiEncoderUNet(
    past_channels=1,
    impassable_channels=1,
    context_channels=3,
    zoom_channels=3
).to(DEVICE)

ckpt_path = "checkpoints/long-strict-w-imgs/best_model.pth"
checkpoint = torch.load(ckpt_path, map_location=DEVICE)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


# cuDNN autotuner (very important for CNNs / UNet)
torch.backends.cudnn.benchmark = True


## ----- Warmup (GPU stabilisation) ----- ##
print("Warming up...")

warm_loader = iter(samples)
for _ in range(10):
    sample = next(warm_loader)
    past, imp, ctx, zoom, target = [
        x.to(DEVICE, non_blocking=True) for x in sample
    ]

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = model(past, imp, ctx, zoom)

torch.cuda.synchronize()


## ----- Benchmark ----- ##
start_time = time.time()
processed = 0

for sample in samples:

    if processed >= NO_SAMPLES:
        break

    past, imp, ctx, zoom, target = [
        x.to(DEVICE, non_blocking=True) for x in sample
    ]

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = model(past, imp, ctx, zoom)

    processed += past.shape[0]   # count real samples


torch.cuda.synchronize()
end_time = time.time()


## ----- Results ----- ##
elapsed = end_time - start_time
samples_per_sec = processed / elapsed
time_per_sample = elapsed / processed

print()
print("-" * 20, f"on {DEVICE}", "-" * 20)
print(f"Processed samples:               {processed}")
print(f"Samples per second:              {samples_per_sec:.2f}")
print(f"Inference time:                  {elapsed:.2f} s")
print(f"Average time per sample:         {time_per_sample:.4f} s")
print("-" * 50)
print()