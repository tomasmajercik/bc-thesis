"""diagnose.py - Run this FIRST"""
import torch
import numpy as np
from training.datasets import PETSDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataset = PETSDataset(scale=1)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

print("="*60)
print("DATASET DIAGNOSTICS")
print("="*60)

# 1. Check target statistics
target_stats = {
    'zero_count': 0,
    'non_zero_count': 0,
    'mean_values': [],
    'max_values': [],
    'min_nonzero': [],
    'num_nonzero_pixels': []
}

for idx in range(min(len(dataset), 100)):  # check first 100
    past, imp, ctx, zoom, target = dataset[idx]
    
    target_np = target.squeeze().numpy()
    
    if target_np.sum() < 1e-6:
        target_stats['zero_count'] += 1
    else:
        target_stats['non_zero_count'] += 1
        target_stats['mean_values'].append(target_np.mean())
        target_stats['max_values'].append(target_np.max())
        
        nonzero_mask = target_np > 0
        if nonzero_mask.any():
            target_stats['min_nonzero'].append(target_np[nonzero_mask].min())
            target_stats['num_nonzero_pixels'].append(nonzero_mask.sum())

print(f"\n📊 GROUND TRUTH STATISTICS (first 100 samples):")
print(f"  Zero samples: {target_stats['zero_count']}")
print(f"  Non-zero samples: {target_stats['non_zero_count']}")

if target_stats['non_zero_count'] > 0:
    print(f"\n  Non-zero heatmaps:")
    print(f"    Mean value: {np.mean(target_stats['mean_values']):.6f}")
    print(f"    Max value: {np.mean(target_stats['max_values']):.6f}")
    print(f"    Min non-zero: {np.mean(target_stats['min_nonzero']):.6f}")
    print(f"    Avg non-zero pixels: {np.mean(target_stats['num_nonzero_pixels']):.1f}")
    print(f"    % of image that's non-zero: {100 * np.mean(target_stats['num_nonzero_pixels']) / (768*576):.2f}%")

# 2. Visualize a few samples
print(f"\n🖼️  VISUALIZING 3 SAMPLES...")
fig, axes = plt.subplots(3, 5, figsize=(15, 9))

for i in range(3):
    past, imp, ctx, zoom, target = dataset[i]
    
    axes[i, 0].imshow(past.squeeze(), cmap='gray')
    axes[i, 0].set_title('Past Traj')
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(imp.squeeze(), cmap='gray')
    axes[i, 1].set_title('Obstacle Mask')
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(ctx.permute(1,2,0).numpy().astype(np.uint8))
    axes[i, 2].set_title('Context')
    axes[i, 2].axis('off')
    
    axes[i, 3].imshow(zoom.permute(1,2,0).numpy().astype(np.uint8))
    axes[i, 3].set_title('Zoom')
    axes[i, 3].axis('off')
    
    axes[i, 4].imshow(target.squeeze(), cmap='hot', vmin=0, vmax=1)
    axes[i, 4].set_title(f'Target (max={target.max():.3f})')
    axes[i, 4].axis('off')

plt.tight_layout()
plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
print(f"  Saved to: dataset_samples.png")

# 3. Check value distribution
print(f"\n📈 TARGET VALUE DISTRIBUTION:")
all_values = []
for idx in range(min(len(dataset), 100)):
    _, _, _, _, target = dataset[idx]
    all_values.extend(target.flatten().numpy())

all_values = np.array(all_values)
print(f"  Min: {all_values.min():.6f}")
print(f"  Max: {all_values.max():.6f}")
print(f"  Mean: {all_values.mean():.6f}")
print(f"  Median: {np.median(all_values):.6f}")
print(f"  % zeros: {100 * (all_values < 1e-6).sum() / len(all_values):.2f}%")

# 4. Check a model forward pass
from model.model import MultiEncoderUNet
model = MultiEncoderUNet(past_channels=1, impassable_channels=1, 
                         context_channels=3, zoom_channels=3)

past, imp, ctx, zoom, target = dataset[0]
with torch.no_grad():
    out = model(past.unsqueeze(0), imp.unsqueeze(0), 
                ctx.unsqueeze(0), zoom.unsqueeze(0))

print(f"\n🤖 MODEL OUTPUT CHECK:")
print(f"  Output shape: {out.shape}")
print(f"  Output min: {out.min().item():.6f}")
print(f"  Output max: {out.max().item():.6f}")
print(f"  Output mean: {out.mean().item():.6f}")
print(f"  Target mean: {target.mean().item():.6f}")
print(f"  Target max: {target.max().item():.6f}")

print("\n" + "="*60)