"""
Modality ablation study
"""
import matplotlib
import torch
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from training.utils import split_ds
from torch.utils.data import DataLoader
from model.model import MultiEncoderUNet
from training.datasets import PETSDataset
from training.losses import NonZeroDiceLoss, SparseHeatmapLoss, MAELoss
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def visualize_results(path):
    df = pd.read_csv(path)

    group_cols = ['past','obstacle','context','zoom']
    agg_df = df.groupby(group_cols)[['sparse_heatmap','nonzero_dice','mae']].mean().reset_index()

    # Baseline: all modalities present
    baseline = agg_df[
        (agg_df.past==1) &
        (agg_df.obstacle==1) &
        (agg_df.context==1) &
        (agg_df.zoom==1)
    ]

    baseline_means = baseline[['sparse_heatmap','nonzero_dice','mae']].iloc[0]

    # Compute delta (Δ) for full table
    for loss in ['sparse_heatmap','nonzero_dice','mae']:
        agg_df[f'{loss}_Δ'] = agg_df[loss] - baseline_means[loss]

    agg_df = agg_df.round(6)

    print("\n=== All Modality Combinations (mean losses + Δ) ===\n")
    print(agg_df.to_string(index=False))

    # === Single-modality importance table ===
    single_ablation = agg_df[
        agg_df[['past','obstacle','context','zoom']].sum(axis=1) == 3
    ]

    importance = {}
    for mod in ['past','obstacle','context','zoom']:
        mod_delta = single_ablation[single_ablation[mod]==0]
        importance[mod] = mod_delta[
            ['sparse_heatmap_Δ','nonzero_dice_Δ','mae_Δ']
        ].mean()

    importance_df = pd.DataFrame(importance).T.round(6)

    importance_df.rename(columns={
        'sparse_heatmap_Δ': 'SparseHeatmap Δ',
        'nonzero_dice_Δ': 'NonZeroDice Δ',
        'mae_Δ': 'MAE Δ'
    }, inplace=True)

    print("\n=== Modality Importance (Single Ablation Δ from Baseline) ===\n")
    print(importance_df.to_string())
def _plot_ablation_bars(csv_path):
    df = pd.read_csv(csv_path)
    df = df.groupby(['past', 'obstacle', 'context', 'zoom']).mean().reset_index()

    modalities = ['past', 'obstacle', 'context', 'zoom']
    metrics = ['sparse_heatmap', 'nonzero_dice', 'mae']
    colors = ['#e74c3c', '#2ecc71']  # red=zeroed, green=kept

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Marginal effect of each modality per metric', fontsize=13)

    x = np.arange(len(modalities))
    width = 0.35

    for ax, metric in zip(axes, metrics):
        for k, (label, color) in enumerate(zip(['zeroed', 'kept'], colors)):
            vals = [df[df[mod] == k][metric].mean() for mod in modalities]
            bars = ax.bar(x + k * width, vals, width, label=label, color=color, alpha=0.85)
            # for bar, val in zip(bars, vals):
            #     ax.text(bar.get_x() + bar.get_width()/2, val * 1.01, f'{val:.3f}',
            #             ha='center', va='bottom', fontsize=7.5, rotation=45)

        ax.set_title(metric, fontsize=11)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(modalities, fontsize=10)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('previews/interpretability/modality_ablation_losses/ablation_bars.png', dpi=150)
def plot_ablation_scatter(csv_path, metric='sparse_heatmap'):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    np.random.seed(42)

    df = pd.read_csv(csv_path)
    df = df.groupby(
        ['past', 'obstacle', 'context', 'zoom']
    )[metric].mean().reset_index()

    modalities = ['past', 'obstacle', 'context', 'zoom']

    # ===============================
    # Baseline
    # ===============================
    baseline = df[
        (df.past == 1) &
        (df.obstacle == 1) &
        (df.context == 1) &
        (df.zoom == 1)
    ][metric].iloc[0]

    df['delta'] = df[metric] - baseline
    df['num_inactive'] = 4 - df[modalities].sum(axis=1)

    # ===============================
    # Label disabled
    # ===============================
    def get_disabled_label(row):
        disabled = []
        if row['past'] == 0:
            disabled.append('P')
        if row['obstacle'] == 0:
            disabled.append('O')
        if row['context'] == 0:
            disabled.append('C')
        if row['zoom'] == 0:
            disabled.append('Z')

        if len(disabled) == 0:
            return "Full"
        return "+".join(disabled)

    df['label'] = df.apply(get_disabled_label, axis=1)

    # ===============================
    # Random vibrant colors
    # ===============================
    unique_labels = df['label'].unique()
    cmap = matplotlib.colormaps['tab20'](np.linspace(0, 1, len(unique_labels)))

    color_dict = {
        label: mcolors.to_hex(cmap[i]) for i, label in enumerate(unique_labels)
        for i, label in enumerate(unique_labels)
    }

    df['color'] = df['label'].map(color_dict)

    # ===============================
    # Plot
    # ===============================
    plt.figure(figsize=(8, 6))

    plt.scatter(
        df['num_inactive'],
        df['delta'],
        s=40 + df['num_inactive'] * 20,
        c=df['color'],
        alpha=0.9
    )

    # ===============================
    # Add linear diagonal line
    # ===============================
    x_line = np.array([0, 4])
    y_line = np.linspace(df['delta'].min(), df['delta'].max(), 2)
    plt.plot(x_line, y_line, 'gray', linestyle='-', linewidth=1.5, alpha=0.6)

    # ===============================
    # Anti-overlap text
    # ===============================
    grouped = df.groupby(['num_inactive', 'delta'])

    for (_, _), group in grouped:
        for i, (_, row) in enumerate(group.iterrows()):

            plt.text(
                row['num_inactive'] + 0.05,
                row['delta'] + np.random.uniform(-0.07, 0.07),  # jitter to reduce overlap
                row['label'],
                fontsize=7,
                color=row['color'],
                weight='bold'
            )

    plt.axhline(0, linestyle='--', color='black', linewidth=1)

    plt.xlabel("Number of Inactive Modalities")
    plt.ylabel(f"Δ {metric} (vs baseline)")
    plt.title(f"Ablation Scatter — {metric}")

    plt.xticks([0,1,2,3,4])
    plt.tight_layout()
    plt.savefig(
        'previews/interpretability/modality_ablation_losses/scatter.png',
        dpi=150
    )

    return df
if __name__ == "__main__":

    plot_ablation_scatter("previews/interpretability/modality_ablation_losses/results.csv")
    exit(0)

    batch_size = 8
    dataset = PETSDataset(scale=0.5)

    train_ds, val_ds = split_ds(.01, dataset)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size, shuffle=False)

    ## Load model
    model = MultiEncoderUNet().to(DEVICE)
    ckpt_path = "checkpoints/long-strict-w-imgs/best_model.pth"
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


    modalities = ['past', 'obstacle', 'context', 'zoom']
    combinations = list(itertools.product([0, 1], repeat=len(modalities)))  # 0=zeroed, 1=keep real
    
    # Define losses
    losses = {
        'sparse_heatmap': SparseHeatmapLoss(),
        'nonzero_dice': NonZeroDiceLoss(),
        'mae': MAELoss()
    }

    results = []

    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Samples"):
            past, obstacle, context, zoom, target = [x.to(DEVICE) for x in batch]

            for combo in combinations:
                past_inp     = past if combo[0] else torch.zeros_like(past)
                obstacle_inp = obstacle if combo[1] else torch.zeros_like(obstacle)
                context_inp  = context if combo[2] else torch.zeros_like(context)
                zoom_inp     = zoom if combo[3] else torch.zeros_like(zoom)

                pred = model(past_inp, obstacle_inp, context_inp, zoom_inp)

                # Compute all losses
                loss_values = {name: loss_fn(pred, target).item() for name, loss_fn in losses.items()}

                results.append({
                    'past': combo[0],
                    'obstacle': combo[1],
                    'context': combo[2],
                    'zoom': combo[3],
                    **loss_values
                })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("previews/interpretability/modality_ablation_losses/results.csv", index=True)
    print("Saved modality ablation results!")

    visualize_results("previews/interpretability/modality_ablation_losses/results.csv")


    ## run with python -m interpretability.ablation_study