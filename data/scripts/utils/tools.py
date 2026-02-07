import cv2
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_params(path="params.yaml"):
    PARAMS_PATH = Path(path)
    with open(PARAMS_PATH, "r") as f:
        params = yaml.safe_load(f)

    INPUT_CFG = params["for_input"]
    GT_CFG = params["for_groundtruth"]

    print("✅ Parameter configuration loaded")
    return INPUT_CFG, GT_CFG

def compose(traj_raster, local_rgb, context_rgb, save_path):
    """
    Merge the four inputs into one numpy array along the channels and save.

    Inputs:
        traj_raster   : (H, W) uint8
        local_rgb     : (h, w, 3) uint8
        context_rgb   : (hc, wc, 3) uint8

    Output:
        channels 0-2   : local_rgb
        channels 3-5   : context_rgb
        channels 6     : traj_raster
    """

    # must be 3D for concatenation
    if traj_raster.ndim == 2:
        traj_raster = np.expand_dims(traj_raster, axis=-1)

    merged = np.concatenate([
        local_rgb, 
        context_rgb, 
        traj_raster
    ], axis=-1).astype(np.uint8)

    # --- save ---
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, merged)

def decompose(original_frame_id, processed_id):
    """
    Unglue the processed 7-channel ndarray and save separate visualizations.
    - channels 0-2: local_rgb
    - channels 3-5: context_rgb
    - channel 6: past_trajectory
    """
    # Define paths
    base_path = Path(__file__).parent.parent.parent / "processed/PETS09"
    raw_path  = Path(__file__).parent.parent.parent / "raw/PETS09"
    save_dir  = Path(__file__).parent.parent.parent / "unglued" / f"{processed_id}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the 7-channel input ndarray
    input_file = base_path / "input" / f"{processed_id}.npy"
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    merged = np.load(input_file) # Shape: (H, W, 7)

    # Slice the channels back into separate modalities
    local_rgb     = merged[:, :, 0:3]
    context_rgb   = merged[:, :, 3:6]
    past_traj     = merged[:, :, 6]    # Grayscale (H, W)


    # Load target (future trajectory heatmap)
    target_file = base_path / "target" / f"{processed_id}.npy"
    future_heatmap = np.load(target_file) if target_file.exists() else None

    # Load original frame for anchor verification
    frame_file = raw_path / "frames" / f"frame_{original_frame_id}.jpg"
    original_frame = cv2.imread(str(frame_file))

    # --- SAVE VISUALIZATIONS ---
    # Local Zoom (Scale 2.0, same pixel coordinates for anchor)
    cv2.imwrite(str(save_dir / "02_local_crop.jpg"), local_rgb)

    # Context View (Scale 1.0)
    cv2.imwrite(str(save_dir / "03_context_crop.jpg"), context_rgb)

    # Past Trajectory (Grayscale to 3-channel for visualization)
    cv2.imwrite(str(save_dir / "04_past_traj.jpg"), past_traj)

    # Future Heatmap (Target)
    if future_heatmap is not None:
        # Convert to uint8 if it's float [0,1]
        heatmap_viz = future_heatmap if future_heatmap.dtype == np.uint8 else (future_heatmap * 255).astype(np.uint8)
        cv2.imwrite(str(save_dir / "05_future_heatmap.jpg"), heatmap_viz)

    # Global Obstacle Mask (Directly from raw)
    mask_file = base_path / "obstacle_mask.npy"
    if mask_file.exists():
        obstacle_mask = np.load(mask_file)
        cv2.imwrite(str(save_dir / "06_global_mask.jpg"), obstacle_mask)
    else:
        print(f"⚠️ Obstacle mask file not found: {mask_file}")

    print(f"✅ Unglued visualizations saved to: {save_dir}")




if __name__ == "__main__":
    decompose(
        original_frame_id="0528",   # original photo
        processed_id="0000",        # processed input
    )
    decompose(
        original_frame_id="0528",   # original photo
        processed_id="0001",        # processed input
    )
    decompose(
        original_frame_id="0528",   # original photo
        processed_id="0002",        # processed input
    )
    decompose(
        original_frame_id="0528",   # original photo
        processed_id="0003",        # processed input
    )
    decompose(
        original_frame_id="0528",   # original photo
        processed_id="0004",        # processed input
    )

