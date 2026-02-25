import yaml
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split

class ConsoleColors:
    ORANGE = "\033[33m"
    GREEN  = "\033[32m"
    RED    = "\033[31m"
    RESET  = "\033[00m"

    INFO = f"{GREEN}[INFO] {RESET}"
    WARN = f"{ORANGE}[WARN] {RESET}"
    ERR  = f"{RED}[ERR] {RESET}"

def np_2_tensor(raw_numpy, device):
    return(
        torch.from_numpy(raw_numpy)
        .permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        .unsqueeze(0)     # (1, H, W, C) - adds batch dim
        .float().to(device)
    )
    
def load_params(path="config/training_cfg.yaml"):
    with open(path, "r") as f:
        CFG = yaml.safe_load(f)

    print(ConsoleColors.INFO + "Config loaded")

    return CFG

def split_ds(train_ratio, dataset):
    train_size = int(train_ratio * len(dataset))
    val_size   = len(dataset) - train_size

    return random_split(dataset, [train_size, val_size])

def split_ds_w_test(train_ratio, dataset, val_ratio=0.1):
    n = len(dataset)

    train_size = int(train_ratio * n)
    val_size   = int(val_ratio * n)
    test_size  = n - train_size - val_size

    return torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

def log_predictions_to_wandb(model, val_loader, epoch, device, num_samples=3):
    """
    Log prediction visualizations to Wandb.
    Shows: Past Traj, Obstacle Mask, Context, Zoom, Ground Truth, Prediction
    """
    model.eval()
    
    # Get one batch
    batch = next(iter(val_loader))
    past, imp, ctx, zoom, target = [x.to(device) for x in batch]
    
    with torch.no_grad():
        pred = model(past, imp, ctx, zoom)
    
    images_to_log = []
    
    for i in range(min(num_samples, past.shape[0])):
        fig, axes = plt.subplots(1, 6, figsize=(18, 3))
        
        # 1. Past Trajectory
        axes[0].imshow(past[i, 0].cpu().numpy(), cmap='gray')
        axes[0].set_title('Past Traj')
        axes[0].axis('off')
        
        # 2. Obstacle Mask
        axes[1].imshow(imp[i, 0].cpu().numpy(), cmap='gray')
        axes[1].set_title('Obstacle Mask')
        axes[1].axis('off')
        
        # 3. Context
        ctx_img = ctx[i].permute(1, 2, 0).cpu().numpy()
        axes[2].imshow(ctx_img)
        axes[2].set_title('Context')
        axes[2].axis('off')
        
        # 4. Zoom
        zoom_img = zoom[i].permute(1, 2, 0).cpu().numpy()
        axes[3].imshow(zoom_img)
        axes[3].set_title('Zoom')
        axes[3].axis('off')
        
        # 5. Ground Truth
        target_img = target[i, 0].cpu().numpy() * 255.0
        axes[4].imshow(target_img, cmap='hot', vmin=0, vmax=255)
        axes[4].set_title(f'GT (max={target_img.max():.2f})')
        axes[4].axis('off')
        
        # 6. Prediction
        pred_img = pred[i, 0].cpu().numpy() * 255.0
        axes[5].imshow(pred_img, cmap='hot', vmin=0, vmax=255)
        axes[5].set_title(f'Pred (max={pred_img.max():.2f}, mean={pred_img.mean():.4f})')
        axes[5].axis('off')
        
        plt.tight_layout()
        
        # Convert plot to image for wandb
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8) # pyright: ignore[reportAttributeAccessIssue]
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_array = img_array[:, :, :3]  # Drop alpha channel, keep RGB
        
        images_to_log.append(wandb.Image(img_array, caption=f"Sample {i+1} - Epoch {epoch}"))
        plt.close(fig)
    
    model.train()
    return images_to_log

if __name__ == "__main__":
    from model.model import MultiEncoderUNet
    from torch.utils.data import DataLoader
    from training.datasets import PETSDataset

    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    dataset     = PETSDataset(scale=0.5)
    val_loader  = DataLoader(dataset, batch_size=1, shuffle=True)

    ## Load model
    model = MultiEncoderUNet(
        past_channels = 1,
        obstacle_channels = 1,
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

    # Generate images using existing function
    images = log_predictions_to_wandb(model, val_loader, epoch=0, device=DEVICE)

    # Save images locally (no function changes)
    for i, img in enumerate(images):
        img.image.save(f"previews/model_io/preview_.png")

# run from root; python -m training.utils