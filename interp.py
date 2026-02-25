import torch
from model.model import MultiEncoderUNet
from training.datasets import PETSDataset
from torch.utils.data import DataLoader, Subset
from interpretability.hooks import FeatureRecorder
from interpretability.visualize import plot_activation_grid
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"; BATCH_SIZE = 1

# sample_indices = [i for i in range(1240,1260)] # up to 4630
sample_indices = [1303] # up to 4630
dataset = PETSDataset(scale=0.5)
subset = Subset(dataset, sample_indices)
samples = DataLoader(
    subset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

model = MultiEncoderUNet().to(DEVICE)
ckpt_path = "checkpoints/long-strict-w-imgs/best_model.pth"
checkpoint = torch.load(ckpt_path, map_location=DEVICE)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

LAYERS_TO_HOOK = {
    "past_encoder_level1": model.past_enc.inc,
    "past_encoder_level2": model.past_enc.down1,
    "past_encoder_level3": model.past_enc.down2,
    
    "obstacle_level1": model.impass_enc.inc,
    "obstacle_level2": model.impass_enc.down1,
    "obstacle_level3": model.impass_enc.down2,
    
    "context_level1": model.ctx_enc.inc,
    "context_level2": model.ctx_enc.down1,
    "context_level3": model.ctx_enc.down2,
    "context_level4": model.ctx_enc.down3,
    
    "zoom_level1": model.zoom_enc.inc,
    "zoom_level2": model.zoom_enc.down1,
    "zoom_level3": model.zoom_enc.down2,
    "zoom_level4": model.zoom_enc.down3,
}


for idx, batch in enumerate(samples):
    with torch.no_grad():
        with FeatureRecorder(LAYERS_TO_HOOK) as hooks: # type: ignore
            past, imp, ctx, zoom, target = [
                x.to(DEVICE, non_blocking=True) for x in batch
            ]
            out = model(past, imp, ctx, zoom)
    
    plot_activation_grid(hooks.activations, ios=(past, imp, ctx, zoom, target, out), mean_across_channels=True, save_path=f"previews/interpretability/encoder_grid_{sample_indices[idx]}.png")

