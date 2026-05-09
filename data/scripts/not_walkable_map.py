""" This script is not used in the final thesis. It was an early attempt to create 
walkable area maps using a segmentation model, but the results were not satisfactory. 
The final thesis uses manual approach instead. """
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# --- Load model once ---
CHECKPOINT = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
processor = SegformerImageProcessor.from_pretrained(CHECKPOINT)
model = SegformerForSemanticSegmentation.from_pretrained(CHECKPOINT).eval()

# --- Obstacle classes ---
OBSTACLE_IDS = [
    3, 4, 5, 6, 7, 8,        # wall–vegetation
    20, 21, 22, 23, 24, 25  # furniture, bins, boxes
]

def extract_ground_points(mask, band_height=10):
    """
    Keeps only the bottom band of each connected component
    to approximate ground-contact regions.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    ground_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 20:
            continue

        y_bottom = y + h
        y_band_start = max(y_bottom - band_height, y)

        component = (labels == i)
        band = np.zeros_like(mask, dtype=bool)
        band[y_band_start:y_bottom, x:x + w] = True

        ground_mask[component & band] = 255

    return ground_mask

def not_walkable_map_segformer(
    image_path,
    output_dir="../raw/PETS09/masks/",
    band_height=12,
):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load image ---
    img = Image.open(image_path).convert("RGB")

    # --- Inference ---
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=img.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    segmentation = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    # --- Full obstacle mask ---
    obstacle_mask = np.isin(segmentation, OBSTACLE_IDS).astype(np.uint8)
    obstacle_mask = cv2.morphologyEx(
        obstacle_mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    obstacle_mask = (obstacle_mask * 255).astype(np.uint8)

    # --- Ground-contact mask ---
    ground_mask = extract_ground_points(obstacle_mask, band_height)

    # --- Save ground-contact mask ONLY ---
    mask_path = output_dir / "obstacle_mask.png"
    Image.fromarray(ground_mask).save(mask_path)

    # --- Visualization ---
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.imshow(obstacle_mask, alpha=0.5, cmap="Reds")
    plt.title("Obstacle Mask")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imshow(ground_mask, alpha=0.8, cmap="Blues")
    plt.title("Ground Contacts")
    plt.axis("off")

    plt.tight_layout()

    viz_path = output_dir / f"{image_path.stem}_map_visualization.png"
    plt.savefig(viz_path, dpi=300)
    plt.close()

    return {
        "ground_contact_mask_path": mask_path,
        "visualization_path": viz_path,
    }


if __name__ == "__main__":
    # Example usage
    results = not_walkable_map_segformer("../raw/PETS09/frames/frame_0000.jpg")
    print(f"Obstacle mask saved at: {results['ground_contact_mask_path']}")
    print(f"Visualization saved at: {results['visualization_path']}")