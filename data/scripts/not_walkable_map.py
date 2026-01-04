# import torch
# import numpy as np
# from PIL import Image
# from pathlib import Path
# import matplotlib.pyplot as plt
# from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# def not_walkable_map(image_path):
#     # --- 1. Load model ---
#     checkpoint = "facebook/mask2former-swin-tiny-cityscapes-semantic"
#     processor = AutoImageProcessor.from_pretrained(checkpoint)
#     model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint).eval()

#     # --- 2. Load image ---
#     img = Image.open(image_path).convert("RGB")

#     # --- 3. Run inference ---
#     inputs = processor(images=img, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)

#     # --- 4. Post-process segmentation map ---
#     segmentation = processor.post_process_semantic_segmentation(
#         outputs, target_sizes=[img.size[::-1]]
#     )[0].cpu().numpy()

#     # --- 5. Cityscapes class labels ---
#     classes = [
#         "road", "sidewalk", "building", "wall", "fence",
#         "pole", "traffic light", "traffic sign", "vegetation", "terrain",
#         "sky", "person", "rider", "car", "truck",
#         "bus", "train", "motorcycle", "bicycle"
#     ]

#     # --- 6. Print all detected classes ---
#     ids, counts = np.unique(segmentation, return_counts=True)
#     print("Detected classes:\n-----------------")
#     for i, c in zip(ids, counts):
#         if i < len(classes):
#             print(f"{i:2d}: {classes[i]:<15} ({c} pixels)")

#     # --- 7. Define obstacle classes ---
#     obstacle_ids = [3, 4, 5, 6, 7, 8]  # wall, fence, pole, traffic light, traffic sign, vegetation

#     # --- 8. Create binary mask for obstacles ---
#     obstacle_mask = np.isin(segmentation, obstacle_ids).astype(np.uint8) * 255  # 0 = background, 255 = obstacle

#    # --- 9. Output directory ---
#     mask_dir = Path("../raw/PETS09/masks/")
#     mask_dir.mkdir(parents=True, exist_ok=True)

#     # --- 10. Save mask ONLY (for future use) ---
#     mask_path = mask_dir / "obstacle_mask.png"
#     Image.fromarray(obstacle_mask).save(mask_path)

#     # --- 11. Visualization ---
#     plt.figure(figsize=(18, 6))

#     # Original image
#     plt.subplot(1, 3, 1)
#     plt.imshow(img)
#     plt.title("Original Image")
#     plt.axis("off")

#     # Obstacle mask only
#     plt.subplot(1, 3, 2)
#     plt.imshow(obstacle_mask, cmap="gray")
#     plt.title("Obstacle Mask")
#     plt.axis("off")

#     # Overlay visualization
#     plt.subplot(1, 3, 3)
#     plt.imshow(img)
#     plt.imshow(obstacle_mask, alpha=0.5, cmap="Reds")
#     plt.title("Detected Obstacles Overlay")
#     plt.axis("off")

#     plt.tight_layout()
#     viz_path = mask_dir / "map_visualization.png"
#     plt.savefig(viz_path, dpi=300)
#     plt.show()

#     print(f"Visualization saved as {viz_path}")
#     print(f"Obstacle mask saved as {mask_path}")
# if __name__ == "__main__":
#     # Example usage
#     not_walkable_map("../raw/PETS09/frames/frame_0000.jpg")


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