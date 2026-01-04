import cv2
import random
from pathlib import Path
import xml.etree.ElementTree as ET

# --- Paths ---
frames_dir = Path("../raw/PETS09/frames/")
xml_path = Path("../raw/PETS09/labels/annotations/PETS2009-S2L1.xml")

# --- Load XML ---
tree = ET.parse(xml_path)
root = tree.getroot()

# Dictionary frame_id -> objects
frames = {}
for frame_node in root.iter("frame"):
    frame_id = int(frame_node.attrib["number"])
    objectlist = frame_node.find("objectlist")
    if objectlist is None:
        continuea
    objs = []
    for obj in objectlist.findall("object"):
        obj_id = int(obj.attrib["id"])
        box = obj.find("box")
        if box is not None:
            x = int(float(box.attrib["xc"]))
            y = int(float(box.attrib["yc"]))
            w = int(float(box.attrib["w"]))
            h = int(float(box.attrib["h"]))
            objs.append((obj_id, x, y, w, h))
    frames[frame_id] = objs

# --- Random colors ---
all_ids = {obj_id for objs in frames.values() for (obj_id, *_ ) in objs}
colors = {obj_id: tuple(random.randint(0,255) for _ in range(3)) for obj_id in all_ids}

# --- FPS + duration ---
FPS = 7
SECONDS = 30
MAX_FRAMES = FPS * SECONDS

# --- Prepare video writer ---
sample_img = cv2.imread(str(frames_dir / f"frame_{min(frames.keys()):04d}.jpg"))
h, w, _ = sample_img.shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("pets_first_30s.mp4", fourcc, FPS, (w, h))

# --- Render & save ---
frame_ids = sorted(frames.keys())[:MAX_FRAMES]

for frame_id in frame_ids:
    jpg = frames_dir / f"frame_{frame_id:04d}.jpg"
    img = cv2.imread(str(jpg))
    if img is None:
        print(f"Frame {frame_id} not found")
        continue

    for obj_id, x, y, w, h in frames[frame_id]:
        color = colors[obj_id]
        top_left = (x - w//2, y - h//2)
        bottom_right = (x + w//2, y + h//2)
        cv2.rectangle(img, top_left, bottom_right, color, 2)
        cv2.putText(img, str(obj_id), (top_left[0], top_left[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(img)

out.release()
print("Saved: pets_first_30s.mp4")
