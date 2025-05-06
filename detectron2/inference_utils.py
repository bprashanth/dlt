"""Utility functions for inference visualization."""

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm


class DetectronVisualizer:
    EDGE_COLOR = 'red'

    def __init__(self, output_dir, class_names):
        """Initialize the DetectronVisualizer.

        @param output_dir: Directory where visualizations and predictions will be saved
        @param class_names: List of class names corresponding to model predictions
        """
        self.output_dir = output_dir
        self.class_names = class_names
        os.makedirs(self.output_dir, exist_ok=True)

    def draw_predictions(self, image, outputs, output_png):
        """Draw instance segmentation predictions on the input image.

        @param image: Input image in BGR format (OpenCV format)
        @param outputs: Detectron2 predictor outputs containing 'instances' with pred_masks, pred_classes, and scores
        @param output_png: Path where the visualization will be saved

        @returns None, saves visualization to output_png path
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictions = outputs["instances"].to("cpu")

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_rgb)
        ax.set_title("Predicted Masks")

        categories_present = set()
        cmap = plt.colormaps["tab20"]

        for i in range(len(predictions)):
            mask = predictions.pred_masks[i].numpy()
            class_id = int(predictions.pred_classes[i])
            score = float(predictions.scores[i])
            color = cmap(class_id * 2)
            categories_present.add(class_id)

            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

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

                border = patches.Polygon(
                    polygon,
                    closed=True,
                    edgecolor=self.EDGE_COLOR,
                    facecolor='none',
                    alpha=1.0,
                    linewidth=1.0
                )
                ax.add_patch(border)

                # Add confidence score text above the polygon
                centroid = np.mean(polygon, axis=0)
                ax.text(centroid[0], centroid[1], f"{score:.2f}",
                        color='white', fontsize=8,
                        bbox=dict(facecolor='black', alpha=0.5,
                                  edgecolor='none', pad=1),
                        ha='center', va='bottom')

        # Simplified legend with only category names
        handles = [
            patches.Patch(
                color=cmap(cid * 2),
                label=f"{self.class_names[cid]}",
                alpha=0.2
            )
            for cid in sorted(categories_present)
        ]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_png, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def save_predictions_json(self, outputs, image_path, output_json):
        """Save predictions in COCO-style JSON format.

        @param outputs: Detectron2 predictor outputs containing 'instances' with pred_masks, pred_classes, and scores
        @param image_path: Path to the original input image (used for image_id in JSON)
        @param output_json: Path where the JSON predictions will be saved

        @returns None, saves predictions to output_json path

        @raises ValueError if no valid segmentations are found
        """
        predictions = outputs["instances"].to("cpu")
        results = []

        for i in range(len(predictions)):
            mask = predictions.pred_masks[i].numpy()
            class_id = int(predictions.pred_classes[i])
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

            result = {
                "image_id": os.path.basename(image_path),
                "category_id": class_id,
                "score": score,
                "segmentation": segmentations
            }
            results.append(result)

        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
