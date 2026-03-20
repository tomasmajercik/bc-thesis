import cv2
import time
import sqlite3
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from utils.tools import compose
from utils.tools import load_params

ORANGE = "\033[33m"
GREEN  = "\033[32m"
RESET  = "\033[00m"
start_time = time.time()

## atrium specific functions
def load_past_traj(db_path):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("""
        SELECT bb.object_id, bb.frame_number, bb.x_top_left, bb.y_top_left, bb.x_bottom_right, bb.y_bottom_right
        FROM bounding_boxes bb
        JOIN objects o ON bb.object_id = o.object_id
        WHERE o.road_user_type = 2
        ORDER BY bb.object_id, bb.frame_number
    """)
    rows = cur.fetchall()
    con.close()

    trajectories = defaultdict(list)
    for pid, frame_id, x1, y1, x2, y2 in rows:
        xc     = (x1 + x2) / 2.0
        yc     = (y1 + y2) / 2.0
        h      = y2 - y1
        x_foot = xc
        y_foot = yc + h / 2.0   # == y2 (bottom center)
        trajectories[pid].append((frame_id, x_foot, y_foot))

    # already ordered by frame_number, but sort just to be safe
    for pid in trajectories:
        trajectories[pid].sort(key=lambda x: x[0])

    return trajectories

def get_people_in_frame(db_path, frame_id):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("""
        SELECT bb.object_id
        FROM bounding_boxes bb
        JOIN objects o ON bb.object_id = o.object_id
        WHERE bb.frame_number = ? AND o.road_user_type = 2
    """, (frame_id,))
    rows = cur.fetchall()
    con.close()

    return [row[0] for row in rows]

def get_bbox(db_path, frame_id, pid):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("""
        SELECT x_top_left, y_top_left, x_bottom_right, y_bottom_right
        FROM bounding_boxes
        WHERE object_id = ? AND frame_number = ?
    """, (pid, frame_id))
    row = cur.fetchone()
    con.close()

    if row is None:
        return None

    x1, y1, x2, y2 = row
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    w  = x2 - x1
    h  = y2 - y1
    return xc, yc, w, h

## ---- everything below is identical to pets_process.py ----

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

        n = len(past)
        intensities = np.linspace(min_intensity, max_intensity, n).astype(np.uint8)

        for i in range(1, n):
            _, x0, y0 = past[i - 1]
            _, x1, y1 = past[i]

            p0 = (int(round(x0)), int(round(y0)))
            p1 = (int(round(x1)), int(round(y1)))

            if (
                0 <= p0[0] < width and 0 <= p0[1] < height and
                0 <= p1[0] < width and 0 <= p1[1] < height
            ):
                cv2.line(raster, p0, p1, color=int(intensities[i]), thickness=radius)

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

        sigma = 2.0 + 3.0 * (i / max(1, n - 1))
        amp   = np.exp(-1.5 * i / max(1, n - 1))
        ksize = int(6 * sigma + 1) | 1

        tmp = np.zeros((height, width), dtype=np.float32)
        tmp[py, px] = amp
        tmp = cv2.GaussianBlur(tmp, (ksize, ksize), sigmaX=sigma)
        heatmap += tmp

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=1.0) # type: ignore
    heatmap /= (heatmap.max() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)

    return heatmap

def get_anchor(bbox):
    xc, yc, w, h = bbox
    return (xc, yc + h / 2.0)

def zoom_n_crop(frame, anchor_point, scale):
    H, W = frame.shape[:2]
    ax, ay = anchor_point

    crop_w = W / scale
    crop_h = H / scale

    ratio_x = ax / W
    ratio_y = ay / H

    x1 = ax - crop_w * ratio_x
    y1 = ay - crop_h * ratio_y
    x2 = x1 + crop_w
    y2 = y1 + crop_h

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

    frames_dir  = Path("../raw/atrium/frames")
    db_path     = Path("../raw/atrium/labels/atrium_gt.sqlite")
    mask_path   = Path("../raw/atrium/masks/obstacle_mask.png")

    # --- config ---
    past_steps    = INPUT_CFG["past_traj_steps"]
    local_scale   = INPUT_CFG["local_scale"]
    context_scale = INPUT_CFG["context_scale"]
    traj_method   = INPUT_CFG["traj_sampling_method"]
    # ================================
    future_steps  = GT_CFG["future_traj_steps"]
    gt_traj_method = GT_CFG["traj_sampling_method"]

    # ================================
    iterator = 0
    frame_ids = sorted([
        int(p.stem)
        for p in frames_dir.glob("*.jpg")
    ])

    trajectories = load_past_traj(db_path)

    for frame_id in tqdm(frame_ids, desc="Processing frames"):
        # --- load frame ---
        img = cv2.imread(str(frames_dir / f"{frame_id:08d}.jpg"))
        H, W, _ = img.shape # type: ignore

        # --- load annotations ---
        people = get_people_in_frame(db_path, frame_id)

        obstacle_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert obstacle_mask is not None, "Obstacle mask not found"

        print(f"{GREEN}[INFO]{RESET} 👉 Frame {frame_id} has people: {people}")

        # ==========================================================
        # OUTPUT CONTAINERS
        # ==========================================================
        traj_rasters    = {}
        local_crops     = {}
        context_crops   = {}
        future_heatmaps = {}

        # ==========================================================
        # MAIN LOOP
        # ==========================================================
        for pid in people:
            traj = trajectories[pid]

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

            bbox = get_bbox(db_path, frame_id, pid)
            if bbox is None:
                print(f"{ORANGE}[Warn]{RESET} pid {pid}: missing bbox")
                continue

            anchor_point = get_anchor(bbox)
            local_rgb    = zoom_n_crop(img, anchor_point, local_scale)
            context_rgb  = zoom_n_crop(img, anchor_point, context_scale)

            heatmap = rasterize_future_traj(
                traj=traj,
                frame_id=frame_id,
                future_steps=future_steps,
                height=H,
                width=W,
                method=gt_traj_method,
            )
            if heatmap is None:
                print(f"{ORANGE}[Info]{RESET} pid {pid}: skipped (not enough future)")
                continue

            traj_rasters[pid]    = raster
            local_crops[pid]     = local_rgb
            context_crops[pid]   = context_rgb
            future_heatmaps[pid] = heatmap

        # ==========================================================
        # compose and save
        # ==========================================================
        for pid in future_heatmaps.keys():
            save_file = Path(f"../processed/atrium/input/{iterator:04d}.npy")
            gt_dir    = Path("../processed/atrium/target"); gt_dir.mkdir(parents=True, exist_ok=True)

            compose(
                traj_raster = traj_rasters[pid],
                local_rgb   = local_crops[pid],
                context_rgb = context_crops[pid],
                save_path   = save_file,
            )

            gt_file = gt_dir / f"{iterator:04d}.npy"
            np.save(gt_file, future_heatmaps[pid])

            coords_dir = Path("../processed/atrium/target_coords")
            coords_dir.mkdir(parents=True, exist_ok=True)

            frames_list  = [f for (f, _, _) in trajectories[pid]]
            idx          = frames_list.index(frame_id)
            future_coords = [(x, y) for (_, x, y) in trajectories[pid][idx + 1 : idx + 1 + future_steps]]

            np.save(coords_dir / f"{iterator:04d}.npy", np.array(future_coords, dtype=np.float32))

            iterator += 1

    obstacle_mask_file = Path("../processed/atrium/obstacle_mask.npy")
    obstacle_mask = obstacle_mask[..., None]   # (H, W, 1)
    np.save(obstacle_mask_file, obstacle_mask)

    print(f"{GREEN}[INFO]{RESET} 👉 Process completed in {(time.time() - start_time):.2f}s")
    print(f"{GREEN}[INFO]{RESET} ✅ Dataset processing done, outputs saved to: {save_file.parent}/*.npy")


# run from `cd data/scripts` with `python atrium_process.py`