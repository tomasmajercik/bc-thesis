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

def glue(traj_raster, local_rgb, context_rgb, obstacle_mask, anchor_xy, save_path):
    """
    Merge the four inputs into one numpy array along the channels and save.

    Inputs:
        traj_raster   : (H, W) uint8
        local_rgb     : (h, w, 3) uint8
        context_rgb   : (hc, wc, 3) uint8
        obstacle_mask : (hm, wm) uint8
        anchor_xy     : (2,) float

    Output:
        saves merged array as .npy at save_path
    """

     # --- ensure arrays are in uint8 ---
    traj_raster   = np.expand_dims(traj_raster, axis=-1)       # H,W -> H,W,1
    obstacle_mask = np.expand_dims(obstacle_mask, axis=-1)     # hm,wm -> hm,wm,1

    # --- stack channels ---
    # merged = np.array([traj_raster, local_rgb, context_rgb, obstacle_mask], dtype=object)
    data = {
        "traj_raster": traj_raster,
        "local_rgb": local_rgb,
        "context_rgb": context_rgb,
        "obstacle_mask": obstacle_mask,
        "anchor_xy": np.array(anchor_xy, dtype=np.float32)
    }

    # --- save ---
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, data, allow_pickle=True)

def unglue(original_frame_id, processed_id):
    """
    Save individual visualizations for a frame and processed ndarray.
    
    Outputs 5 separate images:
        - original frame with anchor
        - obstacle mask
        - local crop
        - context crop
        - past trajectory
        - future heatmap
    """
    base_path = Path(__file__).parent.parent.parent / "processed/PETS09"
    raw_path  = Path(__file__).parent.parent.parent / "raw/PETS09"
    save_dir  = Path(__file__).parent.parent.parent / "unglued" / processed_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- load original frame ---
    frame_file = raw_path / "frames" / f"frame_{original_frame_id}.jpg"
    original_frame = cv2.imread(str(frame_file))
    if original_frame is None:
        raise FileNotFoundError(f"Original frame not found: {frame_file}")

    # --- load processed input ndarray ---
    input_file = base_path / "input" / f"{processed_id}.npy"
    merged = np.load(input_file, allow_pickle=True).item()  # .item() ak je uložený dict
    traj_raster    = merged["traj_raster"]
    local_rgb      = merged["local_rgb"]
    context_rgb    = merged["context_rgb"]
    obstacle_mask  = merged["obstacle_mask"]
    anchor_xy      = merged["anchor_xy"]

    # --- draw anchor on original frame ---
    anchor_frame = original_frame.copy()
    ax, ay = map(int, anchor_xy)
    cv2.circle(anchor_frame, (ax, ay), radius=5, color=(0,0,255), thickness=-1)  # červený bod
    cv2.imwrite(save_dir / "original_with_anchor.jpg", anchor_frame)

    # --- save other images ---
    cv2.imwrite(save_dir / "local_crop.jpg", local_rgb)
    cv2.imwrite(save_dir / "context_crop.jpg", context_rgb)
    cv2.imwrite(save_dir / "obstacle_mask.jpg", obstacle_mask)

    # --- past trajectory ---
    traj_raster_rgb = traj_raster
    if traj_raster_rgb.ndim == 2:
        traj_raster_rgb = cv2.cvtColor(traj_raster_rgb, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(save_dir / "past_traj.jpg", traj_raster_rgb)

    # --- future heatmap ---
    target_file = base_path / "target" / f"{processed_id}.npy"
    future_heatmap = np.load(target_file, allow_pickle=True)
    if future_heatmap.max() > 1.0:
        future_heatmap = future_heatmap / 255.0

    # convert to uint8 grayscale
    heatmap_gray = (future_heatmap * 255).astype(np.uint8)
    heatmap_gray_bgr = cv2.cvtColor(heatmap_gray, cv2.COLOR_GRAY2BGR)  # to save as 3-channel jpg
    cv2.imwrite(save_dir / "future_heatmap.jpg", heatmap_gray_bgr)

    print(f"✅ Saved all unglued images to: {save_dir}")





if __name__ == "__main__":
    unglue(
        original_frame_id="0528",   # original photo
        processed_id="0000",        # processed input
    )
    unglue(
        original_frame_id="0528",   # original photo
        processed_id="0001",        # processed input
    )
    unglue(
        original_frame_id="0528",   # original photo
        processed_id="0002",        # processed input
    )
    unglue(
        original_frame_id="0528",   # original photo
        processed_id="0003",        # processed input
    )
    unglue(
        original_frame_id="0528",   # original photo
        processed_id="0004",        # processed input
    )

