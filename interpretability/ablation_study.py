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
# from training.losses import NonZeroDiceLoss, SparseHeatmapLoss, MAELoss
from training.metrics import EMDMetric, KLDMetric, NSSMetric, FDEMetric, MRMetric
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def visualize_results(path):
    df = pd.read_csv(path)

    metric_cols = ['emd', 'kld', 'nss', 'fde', 'mr']
    group_cols  = ['past', 'obstacle', 'context', 'zoom']

    agg_df = df.groupby(group_cols)[metric_cols].mean().reset_index()

    # Baseline: all modalities present
    baseline = agg_df[
        (agg_df.past==1) &
        (agg_df.obstacle==1) &
        (agg_df.context==1) &
        (agg_df.zoom==1)
    ]
    baseline_means = baseline[metric_cols].iloc[0]

    # Compute delta (Δ) for full table
    for m in metric_cols:
        agg_df[f'{m}_Δ'] = agg_df[m] - baseline_means[m]

    agg_df = agg_df.round(6)

    print("\n=== All Modality Combinations (mean metrics + Δ) ===\n")
    print(agg_df.to_string(index=False))

    # === Single-modality importance table ===
    single_ablation = agg_df[
        agg_df[['past', 'obstacle', 'context', 'zoom']].sum(axis=1) == 3
    ]

    delta_cols = [f'{m}_Δ' for m in metric_cols]

    importance = {}
    for mod in ['past', 'obstacle', 'context', 'zoom']:
        mod_delta = single_ablation[single_ablation[mod] == 0]
        importance[mod] = mod_delta[delta_cols].mean()

    importance_df = pd.DataFrame(importance).T.round(6)
    importance_df.columns = [f'{m.upper()} Δ' for m in metric_cols]

    print("\n=== Modality Importance (Single Ablation Δ from Baseline) ===\n")
    print(importance_df.to_string())
# def _plot_ablation_bars(csv_path):
#     df = pd.read_csv(csv_path)
#     df = df.groupby(['past', 'obstacle', 'context', 'zoom']).mean().reset_index()

#     modalities  = ['past', 'obstacle', 'context', 'zoom']
#     metric_cols = ['emd', 'kld', 'nss', 'fde', 'mr']
#     colors      = ['#e74c3c', '#2ecc71']  # red=zeroed, green=kept

#     fig, axes = plt.subplots(1, len(metric_cols), figsize=(22, 5))
#     fig.suptitle('Marginal effect of each modality per metric', fontsize=13)

#     x     = np.arange(len(modalities))
#     width = 0.35

#     for ax, metric in zip(axes, metric_cols):
#         for k, (label, color) in enumerate(zip(['zeroed', 'kept'], colors)):
#             vals = [df[df[mod] == k][metric].mean() for mod in modalities]
#             ax.bar(x + k * width, vals, width, label=label, color=color, alpha=0.85)

#         ax.set_title(metric.upper(), fontsize=11)
#         ax.set_xticks(x + width / 2)
#         ax.set_xticklabels(modalities, fontsize=10)
#         ax.legend(fontsize=9)

#     plt.tight_layout()
#     plt.savefig('previews/interpretability/modality_ablation_losses/ablation_bars.png', dpi=150)
def _plot_ablation_bars(csv_path):
    """
    Plots modality importance as grouped bar charts, one subplot per metric.
    Each bar = |Δ metric| when that modality is removed (single ablation).
    Each metric has its own y-scale.
    """
    df = pd.read_csv(csv_path)
    df = df.groupby(['past', 'obstacle', 'context', 'zoom']).mean().reset_index()

    metric_cols = ['emd', 'kld', 'nss', 'fde', 'mr']
    modalities  = ['past', 'obstacle', 'context', 'zoom']
    colors      = ['#e05c5c', '#e0955c', '#5c8fe0', '#5cc47a']  # one color per modality

    # --- Compute baseline ---
    baseline = df[
        (df.past == 1) & (df.obstacle == 1) &
        (df.context == 1) & (df.zoom == 1)
    ][metric_cols].iloc[0]

    # --- Single ablation: exactly one modality off ---
    single_ablation = df[df[modalities].sum(axis=1) == 3]

    # --- Build importance matrix: rows=modalities, cols=metrics ---
    importance = np.zeros((len(modalities), len(metric_cols)))

    for i, mod in enumerate(modalities):
        row = single_ablation[single_ablation[mod] == 0]
        if len(row) == 0:
            continue
        for j, metric in enumerate(metric_cols):
            delta = row[metric].values[0] - baseline[metric]
            importance[i, j] = abs(delta)

    # --- Plot ---
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(18, 5))
    fig.suptitle('Modality Importance — |Δ| per Metric (single ablation)', fontsize=13, y=1.02)

    x = np.arange(len(modalities))

    for j, (ax, metric) in enumerate(zip(axes, metric_cols)):
        vals = importance[:, j]
        bars = ax.bar(x, vals, color=colors, alpha=0.88, width=0.6, edgecolor='white', linewidth=0.8)

        # Annotate bar tops
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + max(vals) * 0.02,
                f'{val:.3f}',
                ha='center', va='bottom',
                fontsize=8, color='#333333'
            )

        ax.set_title(metric.upper(), fontsize=12, fontweight='bold', pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in modalities], fontsize=10, rotation=15)
        ax.set_ylabel('|Δ|', fontsize=10)
        ax.set_ylim(0, max(vals) * 1.18 if max(vals) > 0 else 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(
        'previews/interpretability/modality_ablation_losses/ablation_bars.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close()
def _plot_ablation_heatmap(csv_path):
    """
    Plots a normalized modality importance heatmap.
    Each cell = |Δ metric| when that modality is removed (single ablation).
    Values are normalized per metric column to [0, 1] so all metrics
    are on the same scale and visually comparable.
    """
    df = pd.read_csv(csv_path)
    df = df.groupby(['past', 'obstacle', 'context', 'zoom']).mean().reset_index()

    metric_cols = ['emd', 'kld', 'nss', 'fde', 'mr']
    modalities  = ['past', 'obstacle', 'context', 'zoom']

    # --- Compute baseline (all modalities on) ---
    baseline = df[
        (df.past == 1) & (df.obstacle == 1) &
        (df.context == 1) & (df.zoom == 1)
    ][metric_cols].iloc[0]

    # --- Single ablation: exactly one modality off ---
    single_ablation = df[df[modalities].sum(axis=1) == 3]

    # --- Build importance matrix: rows=modalities, cols=metrics ---
    importance = np.zeros((len(modalities), len(metric_cols)))

    for i, mod in enumerate(modalities):
        row = single_ablation[single_ablation[mod] == 0]
        if len(row) == 0:
            continue
        for j, metric in enumerate(metric_cols):
            delta = row[metric].values[0] - baseline[metric]
            importance[i, j] = abs(delta)   # absolute delta = how much removing hurts

    # --- Normalize each column to [0, 1] ---
    col_max = importance.max(axis=0)
    col_max[col_max == 0] = 1  # avoid division by zero
    importance_norm = importance / col_max

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 4))

    im = ax.imshow(importance_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized importance (per metric)', fontsize=10)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['low', 'medium', 'high'])

    # Axis labels
    ax.set_xticks(range(len(metric_cols)))
    ax.set_xticklabels([m.upper() for m in metric_cols], fontsize=11)
    ax.set_yticks(range(len(modalities)))
    ax.set_yticklabels([m.capitalize() for m in modalities], fontsize=11)

    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Modality removed', fontsize=11)
    ax.set_title('Modality Importance — Normalized |Δ| per Metric', fontsize=12, pad=12)

    # Annotate each cell with the raw absolute delta
    for i in range(len(modalities)):
        for j in range(len(metric_cols)):
            raw_val = importance[i, j]
            text_color = 'white' if importance_norm[i, j] > 0.6 else 'black'
            ax.text(j, i, f'{raw_val:.3f}',
                    ha='center', va='center',
                    fontsize=9, color=text_color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(
        'previews/interpretability/modality_ablation_losses/ablation_heatmap.png',
        dpi=150
    )
    plt.close()
def plot_ablation_scatter(csv_path, metric='emd'):
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

    # Baseline
    baseline = df[
        (df.past == 1) &
        (df.obstacle == 1) &
        (df.context == 1) &
        (df.zoom == 1)
    ][metric].iloc[0]

    df['delta']        = df[metric] - baseline
    df['num_inactive'] = 4 - df[modalities].sum(axis=1)

    # Label disabled modalities
    def get_disabled_label(row):
        disabled = []
        if row['past']     == 0: disabled.append('P')
        if row['obstacle'] == 0: disabled.append('O')
        if row['context']  == 0: disabled.append('C')
        if row['zoom']     == 0: disabled.append('Z')
        return "Full" if len(disabled) == 0 else "+".join(disabled)

    df['label'] = df.apply(get_disabled_label, axis=1)

    # Colors
    unique_labels = df['label'].unique()
    cmap = matplotlib.colormaps['tab20'](np.linspace(0, 1, len(unique_labels)))
    color_dict = {label: mcolors.to_hex(cmap[i]) for i, label in enumerate(unique_labels)}
    df['color'] = df['label'].map(color_dict)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        df['num_inactive'],
        df['delta'],
        s=20 + df['num_inactive'] * 20,
        c=df['color'],
        alpha=0.9
    )

    x_line = np.array([0, 4])
    y_line = np.linspace(df['delta'].min(), df['delta'].max(), 2)
    plt.plot(x_line, y_line, 'gray', linestyle='-', linewidth=1.5, alpha=0.6)

    grouped = df.groupby(['num_inactive', 'delta'])
    for (_, _), group in grouped:
        for _, row in group.iterrows():
            plt.text(
                row['num_inactive'] + 0.05,
                row['delta'] + np.random.uniform(-0.01, 0.01),
                row['label'],
                fontsize=7,
                color=row['color'],
                weight='bold'
            )

    plt.axhline(0, linestyle='--', color='black', linewidth=1)
    plt.xlabel("Number of Inactive Modalities")
    plt.ylabel(f"Δ {metric.upper()} (vs baseline)")
    plt.title(f"Ablation Scatter - {metric.upper()}")
    plt.xticks([0, 1, 2, 3, 4])
    plt.tight_layout()
    plt.savefig(
        'previews/interpretability/modality_ablation_losses/scatter.png',
        dpi=150
    )

    return df


if __name__ == "__main__":

    _plot_ablation_bars("previews/interpretability/modality_ablation_losses/results.csv")
    # plot_ablation_scatter("previews/interpretability/modality_ablation_losses/results.csv")
    exit(0)

    batch_size = 8
    dataset = PETSDataset(scale=0.5, return_coords=True)

    train_ds, val_ds = split_ds(.01, dataset)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size, shuffle=False)

    ## Load model
    model = MultiEncoderUNet().to(DEVICE)
    # ckpt_path = "checkpoints/long-strict-w-imgs/best_model.pth"
    ckpt_path = "checkpoints/new-obstacle-mask/best_model.pth"
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


    modalities = ['past', 'obstacle', 'context', 'zoom']
    combinations = list(itertools.product([0, 1], repeat=len(modalities)))  # 0=zeroed, 1=keep real
    
    # # Define losses
    # losses = {
    #     'sparse_heatmap': SparseHeatmapLoss(),
    #     'nonzero_dice': NonZeroDiceLoss(),
    #     'mae': MAELoss()
    # }
    metrics = {
        'emd':  EMDMetric(),
        'kld':  KLDMetric(),
        'nss':  NSSMetric(),
        'fde':  FDEMetric(),
        'mr':   MRMetric(threshold_px=20.0),  # tune this to your px/m ratio
    }

    results = []

    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Samples"):
            past, obstacle, context, zoom, target, coords = [x.to(DEVICE) for x in batch]

            for combo in combinations:
                past_inp     = past if combo[0] else torch.zeros_like(past)
                obstacle_inp = obstacle if combo[1] else torch.zeros_like(obstacle)
                context_inp  = context if combo[2] else torch.zeros_like(context)
                zoom_inp     = zoom if combo[3] else torch.zeros_like(zoom)

                pred = model(past_inp, obstacle_inp, context_inp, zoom_inp)

                # Compute all losses
                # loss_values = {name: loss_fn(pred, target).item() for name, loss_fn in losses.items()}
                # Compute all metrics
                metric_values = {name: m(pred, target, coords).item() for name, m in metrics.items()}

                results.append({
                    'past': combo[0],
                    'obstacle': combo[1],
                    'context': combo[2],
                    'zoom': combo[3],
                    # **loss_values
                    **metric_values
                })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("previews/interpretability/modality_ablation_losses/results.csv", index=True)
    print("Saved modality ablation results!")

    visualize_results("previews/interpretability/modality_ablation_losses/results.csv")


    ## run with python -m interpretability.ablation_study