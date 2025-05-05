import json
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import numpy as np
from PIL import Image


def draw_coco_annotations(image_path, coco_path, output_path="coco_overlay.png"):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    with open(coco_path, 'r') as f:
        coco_data = json.load(f)

    # Map image filename to image_id
    filename = os.path.basename(image_path)
    image_entry = next((img for img in coco_data['images'] if os.path.basename(
        img['file_name']) == filename), None)

    if not image_entry:
        print(f"Image {filename} not found in COCO file.")
        return

    image_id = image_entry['id']
    annotations = [ann for ann in coco_data['annotations']
                   if ann['image_id'] == image_id]

    if not annotations:
        print(f"No annotations found for image_id {image_id}.")
    else:
        print(f"Found {len(annotations)} annotations.")

    # Create color map for categories
    category_colors = {}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    cmap = cm.get_cmap('tab10', len(categories))

    for idx, (cat_id, name) in enumerate(categories.items()):
        category_colors[cat_id] = cmap(idx)

    # Plot image and polygons
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.asarray(image))
    ax.set_title(f"COCO Annotations for {filename}")

    print("Added test shapes")

    for ann in annotations:
        cat_id = ann['category_id']
        color = category_colors.get(cat_id, 'red')

        for seg in ann['segmentation']:
            print(
                f"Segmentation length: {len(seg)}, first few coords: {seg[:10]}")

            if not isinstance(seg, list) or len(seg) < 6:
                continue  # Skip invalid segments

            # Flattened list of coordinates → (x, y) pairs
            # Adjust for 1-indexing in COCO format if needed
            xs = [x - 1 for x in seg[0::2]]  # Subtract 1 from x coordinates
            ys = [y - 1 for y in seg[1::2]]  # Subtract 1 from y coordinates

            if len(xs) < 3 or len(ys) < 3:
                continue  # Skip degenerate polygons

            polygon = list(zip(xs, ys))
            poly_patch = patches.Polygon(
                polygon,
                closed=True,
                edgecolor='red',  # Changed to bright red
                fill=True,
                facecolor=color,  # Use original color for fill
                alpha=0.3,
                linewidth=3  # Increased line width
            )
            ax.add_patch(poly_patch)

            # Log bounding box
            bbox = ann.get('bbox', [])
            if bbox:
                print(f"Polygon bbox: {bbox}")

    # Show legend
    handles = [patches.Patch(color=category_colors[cid], label=name)
               for cid, name in categories.items()]
    ax.legend(handles=handles)

    plt.axis('off')
    plt.tight_layout()
    print("About to save figure...")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved overlay to {output_path}")
    # Optional: try to force the figure to close to free up memory
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay COCO annotations on a PNG image")
    parser.add_argument("--png", required=True, help="Path to the PNG image")
    parser.add_argument("--coco", required=True,
                        help="Path to the COCO annotations JSON file")
    parser.add_argument("--output", default="coco_overlay.png",
                        help="Path to save the output image")
    args = parser.parse_args()

    draw_coco_annotations(args.png, args.coco, args.output)
