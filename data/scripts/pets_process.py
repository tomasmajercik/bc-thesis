import cv2
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict

from utils.tools import compose
from utils.tools import load_params

ORANGE = "\033[33m"
GREEN  = "\033[32m"
RESET  = "\033[00m"
start_time = time.time()

## PETS09 specific functions
def load_past_traj(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    trajectories = defaultdict(list)

    for frame_node in root.iter("frame"):
        frame_id = int(frame_node.attrib["number"])
        objectlist = frame_node.find("objectlist")
        if objectlist is None:
            continue

        for obj in objectlist.findall("object"):
            pid = int(obj.attrib["id"])
            box = obj.find("box")
            if box is None:
                continue

            xc = float(box.attrib["xc"])
            yc = float(box.attrib["yc"])
            h  = float(box.attrib["h"])

            x_foot = xc
            y_foot = yc + h / 2.0

            trajectories[pid].append((frame_id, x_foot, y_foot))
    
    # sort by frame id
    for pid in trajectories:
        trajectories[pid].sort(key=lambda x: x[0])

    return trajectories
def get_people_in_frame(xml_path, frame_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for frame_node in root.iter("frame"):
        if int(frame_node.attrib["number"]) == frame_id:
            objectlist = frame_node.find("objectlist")
            if objectlist is None:
                return []

            return [int(obj.attrib["id"]) for obj in objectlist.findall("object")]

    return []

def rasterize_past_traj(
    traj,
    frame_id,
    past_steps,
    height,
    width,
    traj_method,
    radius=2,
    min_intensity=50,
    max_intensity=255,
):
    """
    Rasterize past trajectory as a connected polyline with temporal fading.
    One raster = one pedestrian.
    """
    if traj_method == "linear":
        raster = np.zeros((height, width), dtype=np.uint8)

        frames = [f for (f, _, _) in traj]
        if frame_id not in frames:
            return None

        idx = frames.index(frame_id)

        start = max(0, idx - past_steps)
        past = traj[start:idx]

        if len(past) < 1:
            return None

        # Normalize temporal weights
        n = len(past)
        intensities = np.linspace(min_intensity, max_intensity, n).astype(np.uint8)

        # Draw connected lines
        for i in range(1, n):
            _, x0, y0 = past[i - 1]
            _, x1, y1 = past[i]

            p0 = (int(round(x0)), int(round(y0)))
            p1 = (int(round(x1)), int(round(y1)))

            if (
                0 <= p0[0] < width and 0 <= p0[1] < height and
                0 <= p1[0] < width and 0 <= p1[1] < height
            ):
                cv2.line(
                    raster,
                    p0,
                    p1,
                    color=int(intensities[i]),
                    thickness=radius,
                )

        # Emphasize last position (most recent)
        _, x_last, y_last = past[-1]
        cv2.circle(
            raster,
            (int(round(x_last)), int(round(y_last))),
            radius + 1,
            max_intensity,
            -1,
        )

        return raster
    else:
        raise NotImplementedError(f"Trajectory method '{traj_method}' not implemented.")
def rasterize_future_traj(traj, frame_id, future_steps, height, width, method):
    """
    Rasterize future trajectory as a heatmap using Gaussian blobs.
    The further in the future, the more diffused the point.

    Returns a float32 heatmap in [0,1].
    """
    if method != "gaussian":
        raise NotImplementedError

    heatmap = np.zeros((height, width), dtype=np.float32)

    frames = [f for (f, _, _) in traj]
    if frame_id not in frames:
        return None

    idx = frames.index(frame_id)
    future = traj[idx + 1 : idx + 1 + future_steps]
    if len(future) == 0:
        return None

    n = len(future)

    for i, (_, x, y) in enumerate(future):
        px, py = int(round(x)), int(round(y))
        if not (0 <= px < width and 0 <= py < height):
            continue

        sigma = 2.0 + 3.0 * (i / max(1, n - 1))          # later → wider
        amp   = np.exp(-1.5 * i / max(1, n - 1))        # later → weaker

        ksize = int(6 * sigma + 1) | 1

        tmp = np.zeros((height, width), dtype=np.float32)
        tmp[py, px] = amp
        tmp = cv2.GaussianBlur(tmp, (ksize, ksize), sigmaX=sigma)

        heatmap += tmp

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=1.0)

    # --- normalize to [0,255] ---
    heatmap /= (heatmap.max() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)

    return heatmap
def get_bbox(xml_path, frame_id, pid):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for frame_node in root.iter("frame"):
        if int(frame_node.attrib["number"]) != frame_id:
            continue

        objectlist = frame_node.find("objectlist")
        if objectlist is None:
            return None

        for obj in objectlist.findall("object"):
            if int(obj.attrib["id"]) == pid:
                box = obj.find("box")
                if box is None:
                    return None

                xc = float(box.attrib["xc"])
                yc = float(box.attrib["yc"])
                w  = float(box.attrib["w"])
                h  = float(box.attrib["h"])

                return xc, yc, w, h

    return None
def get_anchor(bbox):
    xc, yc, w, h = bbox
    anchor_x = xc
    anchor_y = yc + h / 2.0 # bottom center of bbox
    return (anchor_x, anchor_y) 
def zoom_n_crop(frame, anchor_point, scale):
    """
    Zoom into image with respect to anchor point (anchor point stays on the same position). 
    """
    H, W = frame.shape[:2]
    ax, ay = anchor_point

    # Calculate cropping box size
    crop_w = W / scale
    crop_h = H / scale

    # Anchor point ratio in original image
    ratio_x = ax / W
    ratio_y = ay / H

    # Top-left corner of crop to maintain anchor at same ratio
    x1 = ax - crop_w * ratio_x
    y1 = ay - crop_h * ratio_y

    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Round and clamp
    x1, y1 = int(round(x1)), int(round(y1))
    x2, y2 = int(round(x2)), int(round(y2))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x2, W)
    y2 = min(y2, H)

    cropped_part = frame[y1:y2, x1:x2]
    return cv2.resize(cropped_part, (W, H), interpolation=cv2.INTER_LINEAR)

if __name__ == "__main__":
    INPUT_CFG, GT_CFG = load_params("configs/params.yaml")

    frames_dir  = Path("../raw/PETS09/frames")
    xml_path    = Path("../raw/PETS09/labels/annotations/PETS2009-S2L1.xml")
    mask_path   = Path("../raw/PETS09/masks/obstacle_mask.png")

    # --- config ---
    past_steps              = INPUT_CFG["past_traj_steps"]
    local_scale             = INPUT_CFG["local_scale"]
    context_scale           = INPUT_CFG["context_scale"]
    traj_method             = INPUT_CFG["traj_sampling_method"]
    # ================================
    future_steps            = GT_CFG["future_traj_steps"]
    gt_traj_method          = GT_CFG["traj_sampling_method"]
    
    # ================================
    iterator = 0
    frame_ids = sorted([
        int(p.stem.split("_")[1])
        for p in frames_dir.glob("frame_*.jpg") # full run
        # for p in frames_dir.glob("frame_0528.jpg") # DEBUG
    ])


    trajectories    = load_past_traj(xml_path)
    for frame_id in tqdm(frame_ids, desc="Processing frames"):
        # --- load frame ---
        img     = cv2.imread(str(frames_dir / f"frame_{frame_id:04d}.jpg"))
        H, W, _ = img.shape

        # --- load annotations ---
        people          = get_people_in_frame(xml_path, frame_id)
        obstacle_mask   = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert obstacle_mask is not None, "Obstacle mask not found"

        print(f"{GREEN}[INFO]{RESET}👉 Frame {frame_id} has people: {people}")

        # ==========================================================
        # OUTPUT CONTAINERS (this is what compose() will later consume)
        # ==========================================================
        traj_rasters    = {}    # pid -> (H, W) uint8
        local_crops     = {}    # pid -> (h, w, 3)
        context_crops   = {}    # pid -> (hc, wc, 3)
        future_heatmaps = {}    # pid -> (H, W) float32

        # ==========================================================
        # MAIN LOOP
        # ==========================================================
        for pid in people:
            traj = trajectories[pid]

            # --- trajectory raster (FULL FRAME, one pedestrian) ---
            raster = rasterize_past_traj(
                traj,
                frame_id=frame_id,
                past_steps=past_steps,
                height=H,
                width=W,
                traj_method=traj_method,
                radius=2,
            )

            if raster is None:
                print(f"{ORANGE}[Info]{RESET} pid {pid}: skipped (not enough past)")
                continue

            # --- bounding box (for crop center) ---
            bbox = get_bbox(xml_path, frame_id, pid)
            if bbox is None:
                print(f"{ORANGE}[Warn]{RESET} pid {pid}: missing bbox")
                continue

            anchor_point = get_anchor(bbox)
            local_rgb    = zoom_n_crop(img, anchor_point, local_scale)
            context_rgb  = zoom_n_crop(img, anchor_point, context_scale)

            # --- future trajectory heatmap ---
            heatmap = rasterize_future_traj(
                traj=traj,
                frame_id=frame_id,
                future_steps=future_steps,
                height=H,
                width=W,
                method=gt_traj_method
            )
            if heatmap is None:
                print(f"{ORANGE}[Info]{RESET} pid {pid}: skipped (not enough future)")
                continue

            # --- store ---
            traj_rasters[pid]    = raster
            local_crops[pid]     = local_rgb
            context_crops[pid]   = context_rgb
            future_heatmaps[pid] = heatmap

        # ==========================================================
        # compose all together and save to a ndarray
        # ==========================================================
        # for pid in traj_rasters.keys():
        for pid in future_heatmaps.keys():
            save_file   = Path(f"../processed/PETS09/input/{iterator:04d}.npy")
            gt_dir      = Path("../processed/PETS09/target"); gt_dir.mkdir(parents=True, exist_ok=True)

            compose(
                traj_raster   = traj_rasters[pid],
                local_rgb     = local_crops[pid],
                context_rgb   = context_crops[pid],
                save_path     = save_file
            )

            gt_file = gt_dir / f"{iterator:04d}.npy"
            np.save(gt_file, future_heatmaps[pid])

            ## also save the coordinates for later evaluation
            coords_dir = Path("../processed/PETS09/target_coords")
            coords_dir.mkdir(parents=True, exist_ok=True)

            frames_list = [f for (f, _, _) in trajectories[pid]]
            idx = frames_list.index(frame_id)
            future_coords = [(x, y) for (_, x, y) in trajectories[pid][idx + 1 : idx + 1 + future_steps]]

            np.save(coords_dir / f"{iterator:04d}.npy", np.array(future_coords, dtype=np.float32))

            iterator += 1
    
    obstacle_mask_file = Path("../processed/PETS09/obstacle_mask.npy")
    obstacle_mask = obstacle_mask[..., None]   # (H, W, 1)
    np.save(obstacle_mask_file, obstacle_mask)

    print(f"{GREEN}[INFO]{RESET} 👉 Process completed in {(time.time() - start_time):.2f}s")
    print(f"{GREEN}[INFO]{RESET} ✅ Dataset processing done, outputs saved to: {save_file.parent}/*.npy")


# run from `cd data/scripts` with `python pets_process.py`