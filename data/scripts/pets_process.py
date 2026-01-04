import cv2
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import defaultdict

from utils.tools import glue
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

        # --- THIS is the tear logic ---
        sigma = 2.0 + 3.0 * (i / max(1, n - 1))          # later → wider
        amp   = np.exp(-1.5 * i / max(1, n - 1))        # later → weaker

        ksize = int(6 * sigma + 1) | 1

        tmp = np.zeros((height, width), dtype=np.float32)
        tmp[py, px] = amp
        tmp = cv2.GaussianBlur(tmp, (ksize, ksize), sigmaX=sigma)

        heatmap += tmp

    # --- very light final smoothing (optional but recommended) ---
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
def bbox_crop(img, xc, yc, bw, bh, pad_w, pad_h):
    """
    Crop image using bounding box + padding.
    Pads with zeros if out of bounds.
    """
    H, W = img.shape[:2]

    x1 = int(round(xc - bw / 2 - pad_w))
    y1 = int(round(yc - bh / 2 - pad_h))
    x2 = int(round(xc + bw / 2 + pad_w))
    y2 = int(round(yc + bh / 2 + pad_h))

    crop_w = x2 - x1
    crop_h = y2 - y1

    crop = np.zeros((crop_h, crop_w, *img.shape[2:]), dtype=img.dtype)

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(W, x2)
    src_y2 = min(H, y2)

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    crop[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

    return crop




if __name__ == "__main__":
    INPUT_CFG, GT_CFG = load_params("utils/params.yaml")

    frames_dir  = Path("../raw/PETS09/frames")
    xml_path    = Path("../raw/PETS09/labels/annotations/PETS2009-S2L1.xml")
    mask_path   = Path("../raw/PETS09/masks/obstacle_mask.png")

    # --- config ---
    past_steps              = INPUT_CFG["past_traj_steps"]
    crop_w, crop_h          = INPUT_CFG["crop_size"]
    ctx_w, ctx_h            = INPUT_CFG["context_size"]
    mask_pad_w, mask_pad_h  = INPUT_CFG["obstacle_mask_size"]
    traj_method             = INPUT_CFG["traj_sampling_method"]
    # ================================
    future_steps            = GT_CFG["future_traj_steps"]
    gt_traj_method          = GT_CFG["traj_sampling_method"]
    # ================================
    iterator = 0
    frame_ids = sorted([
        int(p.stem.split("_")[1])
        # for p in frames_dir.glob("frame_*.jpg")
        for p in frames_dir.glob("frame_0020.jpg") # DEBUG
    ])


    for frame_id in tqdm(frame_ids, desc="Processing frames"):
        # --- load frame ---
        img     = cv2.imread(str(frames_dir / f"frame_{frame_id:04d}.jpg"))
        H, W, _ = img.shape

        # --- load annotations ---
        trajectories    = load_past_traj(xml_path)
        people          = get_people_in_frame(xml_path, frame_id)
        obstacle_mask   = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert obstacle_mask is not None, "Obstacle mask not found"

        print(f"{GREEN}[INFO]{RESET}👉 Frame {frame_id} has people: {people}")

        # ==========================================================
        # OUTPUT CONTAINERS (this is what glue() will later consume)
        # ==========================================================
        traj_rasters   = {}    # pid -> (H, W) uint8
        local_crops    = {}    # pid -> (h, w, 3)
        context_crops  = {}    # pid -> (hc, wc, 3)
        obstacle_crops = {}    # pid -> (hm, wm) uint8
        future_heatmaps = {}  # pid -> (H, W) float32

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
                print(f"{ORANGE}[Warn]{RESET}pid {pid}: skipped (not enough past)")
                continue

            # --- bounding box (for crop center) ---
            bbox = get_bbox(xml_path, frame_id, pid)
            if bbox is None:
                print(f"{ORANGE}[Warn]{RESET}pid {pid}: missing bbox")
                continue

            xc, yc, bw, bh = bbox

            # --- crops ---
            pad_w, pad_h = crop_w, crop_h
            ctx_pad_w, ctx_pad_h = ctx_w, ctx_h

            local_rgb   = bbox_crop(img, xc, yc, bw, bh, pad_w=pad_w, pad_h=pad_h)
            context_rgb = bbox_crop(img, xc, yc, bw, bh, pad_w=ctx_pad_w, pad_h=ctx_pad_h)
            mask_crop   = bbox_crop(obstacle_mask, xc, yc, bw, bh, pad_w=mask_pad_w, pad_h=mask_pad_h)

            # --- future trajectory heatmap ---
            heatmap = rasterize_future_traj(
                traj=traj,
                frame_id=frame_id,
                future_steps=future_steps,
                height=H,
                width=W,
                method=gt_traj_method
            )


            # --- store ---
            traj_rasters[pid]    = raster
            local_crops[pid]     = local_rgb
            context_crops[pid]   = context_rgb
            obstacle_crops[pid]  = mask_crop
            future_heatmaps[pid] = heatmap


        # ==========================================================
        # Glue all together and save to a ndarray
        # ==========================================================
        for pid in traj_rasters.keys():
            save_file   = Path(f"../processed/PETS09/input/{iterator:04d}.npy")
            gt_dir      = Path("../processed/PETS09/target"); gt_dir.mkdir(parents=True, exist_ok=True)

            glue(
                traj_raster   = traj_rasters[pid],
                local_rgb     = local_crops[pid],
                context_rgb   = context_crops[pid],
                obstacle_mask = obstacle_crops[pid],
                save_path     = save_file
            )

            gt_file = gt_dir / f"{iterator:04d}.npy"
            np.save(gt_file, future_heatmaps[pid])

            iterator += 1


    import warnings
    warnings.warn(
        f"\n{ORANGE}[Warn]{RESET}Using allow_pickle=True, maybe you want to change that later "
        "to pad the images? Now it is wrapped in an object array",
        UserWarning
    )
    print(f"{GREEN}[INFO]{RESET} 👉 Process completed in {(time.time() - start_time):.2f}s")
    print(f"{GREEN}[INFO]{RESET} ✅ Dataset processing done, outputs saved to: {save_file.parent}/*.npy")



    """
    3. veci ostavaju:
        ✅ 1. implement unglue() ✅
        ✅ 2. pre vsetky obrazky ✅
        ✅ 3. tqdm ✅
        4. .gitignore and git 
    """