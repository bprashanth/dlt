"""Utility functions for inference visualization."""

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from io import BytesIO
from PIL import Image
import logging
import tempfile


class DetectronVisualizer:
    EDGE_COLOR = 'red'

    def __init__(self, class_names, selected_classes=None, minimal_visualization=False, no_fill=False):
        """Initialize the DetectronVisualizer.

        @param output_dir: Directory where visualizations and predictions will be saved
        @param class_names: List of class names corresponding to model predictions
        @param selected_classes: Optional list of class names to filter
        @param minimal_visualization: If True, only draws polygons without legends, borders, or text
        @param no_fill: If True, polygons will not be filled (only borders will be drawn)
        """
        self.class_names = class_names
        self.logger = logging.getLogger("visualizer")
        self.selected_classes = selected_classes
        self.minimal_visualization = minimal_visualization
        self.no_fill = no_fill

    def _merge_overlapping_contours(self, contours, image_shape):
        """Merge overlapping or intersecting contours into a single continuous border.

        @param contours: List of contours to merge
        @param image_shape: Shape of the image (height, width)
        @returns: List of merged contours
        """
        if not contours:
            return []

        # Create mask with actual image dimensions
        height, width = image_shape[:2]
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        # Draw all contours onto the mask
        for contour in contours:
            cv2.drawContours(combined_mask, [contour], -1, 1, -1)

        # Find all contours in the combined mask
        all_contours, hierarchy = cv2.findContours(
            combined_mask,
            cv2.RETR_TREE,  # Use TREE to get parent-child relationships
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter to keep only the outermost contours
        outer_contours = []
        if hierarchy is not None:
            hierarchy = hierarchy[0]  # Get the first hierarchy level
            for i, (contour, h) in enumerate(zip(all_contours, hierarchy)):
                # If this contour has no parent (is outermost) or its parent is -1
                if h[3] == -1:  # h[3] is the parent index
                    outer_contours.append(contour)
        else:
            # If no hierarchy, use all contours
            outer_contours = all_contours

        return outer_contours

    def render_predictions(self, image, outputs, save_path=None):
        """
        Render predictions over image as a PNG or return as a NumPy array.

        @param image: Input image (OpenCV BGR)
        @param outputs: Detectron2 outputs
        @param save_path: If provided, saves to disk. If None, returns RGB NumPy array

        @returns: None if saved to disk, else NumPy RGB image
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # Use actual image dimensions
        dpi = 100
        fig, ax = plt.subplots(
            figsize=(width/dpi, height/dpi), dpi=dpi, frameon=False)
        ax.set_axis_off()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        ax.imshow(image_rgb)

        categories_present = set()
        cmap = plt.colormaps["tab20"]

        # Dictionary to store contours by class
        class_contours = {}

        predictions = outputs["instances"].to("cpu")
        for i in range(len(predictions)):
            class_id = int(predictions.pred_classes[i])
            class_name = self.class_names[class_id]

            if self.selected_classes and class_name not in self.selected_classes:
                self.logger.info(
                    f"[render_predictions] skipped class {class_name} because it's not in selected_classes")
                continue

            mask = predictions.pred_masks[i].numpy()
            score = float(predictions.scores[i])
            color = cmap(class_id * 2)
            categories_present.add(class_id)

            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Store contours for this class
            if class_id not in class_contours:
                class_contours[class_id] = []
            class_contours[class_id].extend(contours)

            # Draw filled polygons (only if no_fill is False)
            if not self.no_fill:
                for contour in contours:
                    polygon = contour[:, 0, :]
                    patch = patches.Polygon(
                        polygon,
                        closed=True,
                        edgecolor='none',
                        facecolor=color,
                        alpha=0.2,
                        linewidth=0
                    )
                    ax.add_patch(patch)

            # Only add text (confidence score) if not in minimal mode
            if not self.minimal_visualization:
                for contour in contours:
                    polygon = contour[:, 0, :]
                    centroid = np.mean(polygon, axis=0)
                    ax.text(centroid[0], centroid[1], f"{score:.2f}",
                            color='white', fontsize=8,
                            bbox=dict(facecolor='black', alpha=0.5,
                                      edgecolor='none', pad=1),
                            ha='center', va='bottom')

        # Process borders for each class separately
        for class_id, contours in class_contours.items():
            color = cmap(class_id * 2)
            merged_contours = self._merge_overlapping_contours(
                contours, image.shape)
            for contour in merged_contours:
                polygon = contour[:, 0, :]
                border = patches.Polygon(
                    polygon,
                    closed=True,
                    edgecolor=color,  # Use class color for border
                    facecolor='none',
                    alpha=1.0,
                    linewidth=3.0
                )
                ax.add_patch(border)

        # Only add legend if not in minimal mode
        if not self.minimal_visualization:
            handles = [
                patches.Patch(color=cmap(cid * 2),
                              label=self.class_names[cid],
                              alpha=0.2)
                for cid in sorted(categories_present)
            ]
            ax.legend(handles=handles, bbox_to_anchor=(
                1.05, 1), loc='upper left')
            ax.axis('off')
            plt.tight_layout()

        # Use temporary file if save_path not provided
        temp_file = None
        output_path = save_path
        if not save_path:
            temp_file = tempfile.NamedTemporaryFile(suffix='.png')
            output_path = temp_file.name

        try:
            # Save with no padding if minimal visualization
            if self.minimal_visualization:
                plt.savefig(output_path, bbox_inches=None,
                            pad_inches=0, format="png")
            else:
                plt.savefig(output_path, bbox_inches='tight',
                            pad_inches=0, format="png")
            plt.close(fig)
            return cv2.imread(output_path)
        finally:
            if temp_file:
                temp_file.close()

    def format_predictions(self, outputs, image_path, save_path=None):
        """
        Format predictions into COCO-style JSON.

        @param outputs: Detectron2 outputs
        @param image_path: Original image filename
        @param save_path: If provided, writes JSON to disk. Else, returns list.

        @returns: None if saved to disk, else list of prediction dicts
        """
        predictions = outputs["instances"].to("cpu")
        results = []

        for i in range(len(predictions)):
            class_id = int(predictions.pred_classes[i])
            class_name = self.class_names[class_id]
            if self.selected_classes and class_name not in self.selected_classes:
                self.logger.info(
                    f"[format_predictions] would have skipped class {class_name} because it's not in selected_classes")

            mask = predictions.pred_masks[i].numpy()
            score = float(predictions.scores[i])

            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            segmentations = []
            for contour in contours:
                flattened = contour[:, 0, :].flatten().tolist()
                if len(flattened) >= 6:
                    segmentations.append(flattened)

            results.append({
                "image_id": os.path.basename(image_path),
                "category_id": class_id,
                "category_name": class_name,
                "score": score,
                "segmentation": segmentations
            })

        if save_path:
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)
            return None
        else:
            return results

    def draw_predictions(self, image, outputs, output_png):
        self.render_predictions(
            image, outputs, save_path=output_png)

    def get_overlay(self, image, outputs):
        return self.render_predictions(image, outputs, save_path=None)

    def save_predictions_json(self, outputs, image_path, output_json):
        self.format_predictions(outputs, image_path, save_path=output_json)

    def format_predictions_as_json(self, outputs, image_path):
        return self.format_predictions(outputs, image_path, save_path=None)

    def draw_coco_annotations(
            self, image_path, coco_path, output_path="coco_overlay.png"):
        """Draws COCO annotations on an image.

        Assumes coco_path is a COCO JSON file with annotations for the image
        specified by image_path.    

        @param image_path: Path to the image file
        @param coco_path: Path to the COCO JSON file
        @param output_path: Path to save the output image 

        @returns: NumPy RGB image
        """
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
            self.logger.info(f"No annotations found for image_id {image_id}.")
        else:
            self.logger.info(f"Found {len(annotations)} annotations.")

        # Create color map for categories
        category_colors = {}
        categories = {cat['id']: cat['name']
                      for cat in coco_data['categories']}
        cmap = cm.get_cmap('tab10', len(categories))

        for idx, (cat_id, name) in enumerate(categories.items()):
            category_colors[cat_id] = cmap(idx)

        # Plot image and polygons
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(np.asarray(image))
        ax.set_title(f"COCO Annotations for {filename}")

        # Track which categories are actually present in the image
        categories_present = set()

        for ann in annotations:
            cat_id = ann['category_id']
            color = category_colors.get(cat_id, 'red')
            categories_present.add(cat_id)

            for seg in ann['segmentation']:
                if not isinstance(seg, list) or len(seg) < 6:
                    continue  # Skip invalid segments

                # Flattened list of coordinates → (x, y) pairs
                # Subtract 1 from x coordinates
                xs = [x - 1 for x in seg[0::2]]
                # Subtract 1 from y coordinates
                ys = [y - 1 for y in seg[1::2]]

                if len(xs) < 3 or len(ys) < 3:
                    continue  # Skip degenerate polygons

                polygon = list(zip(xs, ys))
                poly_patch = patches.Polygon(
                    polygon,
                    closed=True,
                    edgecolor=color,
                    fill=True,
                    facecolor=color,
                    alpha=0.3,
                    linewidth=3
                )
                ax.add_patch(poly_patch)

        # Show legend only for categories present in the image
        handles = [patches.Patch(color=category_colors[cid], label=categories[cid])
                   for cid in categories_present]
        ax.legend(handles=handles)

        plt.axis('off')
        plt.tight_layout()

        # Use the unified saving logic
        temp_file = None
        save_path = output_path
        if not save_path:
            temp_file = tempfile.NamedTemporaryFile(suffix='.png')
            save_path = temp_file.name

        try:
            plt.savefig(save_path, bbox_inches=None,
                        pad_inches=0, format="png")
            plt.close(fig)
            return cv2.imread(save_path)
        finally:
            if temp_file:
                temp_file.close()
