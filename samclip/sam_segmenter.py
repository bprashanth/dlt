import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import logging
from tqdm import tqdm
from utils import get_color


class SAMSegmenter:

    def __init__(self, model_type="vit_b", checkpoint_path="segment-anything/sam_vit_b_01ec64.pth", device=None, min_area_ratio=0.01):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.min_area_ratio = min_area_ratio

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            min_mask_region_area=0
        )
        self.predictor = SamPredictor(sam)

        self.logger = logging.getLogger('SAMSegmenter')
        self.logger.info(
            f"Initialized SAM with model type: {model_type} on device: {self.device}. Using min_area_ratio: {self.min_area_ratio}.")

    def generate_masks(self, image_rgb):
        self.logger.info("Starting mask generation")
        try:
            h, w = image_rgb.shape[:2]
            min_area = self.min_area_ratio * h * w
            self.mask_generator.min_mask_region_area = int(min_area)
            self.logger.info(
                f"Generating masks with min_area: {int(min_area)}")

            masks = self.mask_generator.generate(image_rgb)
            # TODO(prashanth@): Log how many are < min_area_ratio?
            self.logger.info(f"Generated {len(masks)} masks")
            return masks
        except Exception as e:
            self.logger.error(f"Error generating masks: {e}")
            return []

    def apply_masks(
            self, image_rgb, masks, output_dir="output", min_area_ratio=0.01):

        self.logger.info(
            f"Starting mask application with min_area_ratio: {min_area_ratio}")

        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        h, w = image_rgb.shape[:2]
        min_area = min_area_ratio * h * w

        valid_masks = [m for m in masks if m['area'] >= min_area]
        self.logger.info(
            f"Processing {len(valid_masks)} valid masks out of a total of {len(masks)}")

        # Output each segment as a separate image with white background in the
        # non-segmented areas
        for i, m in tqdm(
            enumerate(valid_masks),
            total=len(valid_masks),
            desc="Processing segments"
        ):

            seg_masks = m["segmentation"]
            masked_image = image_rgb.copy()
            masked_image[~seg_masks] = [255, 255, 255]
            filename = os.path.join(output_dir, f"mask_segment_{i+1}.png")

            # Cropping by bounding box
            x, y, w, h = m['bbox']
            crop = masked_image[y:y+h, x:x+w]

            plt.imsave(filename, crop)
            output_paths.append(filename)

            self.logger.debug(f"Processed segment {i+1} of {len(valid_masks)}")

        self.logger.info("Generating overlay visualizations")
        # Create a final overlay with all segments colored differently
        overlay = image_rgb.copy()
        fig, ax = plt.subplots(figsize=(10, 10))

        # First show the original image
        ax.imshow(overlay)
        legend_patches = []

        for i, m in tqdm(
            enumerate(valid_masks),
            total=len(valid_masks),
            desc="Processing overlay"
        ):
            # Pick a random color for each segment
            color = get_color()

            # Apply a zero mask over the image, then apply the color over the
            # segment area
            seg_mask = m["segmentation"]

            # Make non-segment areas transparent
            mask_rgba = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 4))

            # Over the segment area, apply a transparent color
            mask_rgba[seg_mask, 0] = color[0]
            mask_rgba[seg_mask, 1] = color[1]
            mask_rgba[seg_mask, 2] = color[2]
            mask_rgba[seg_mask, 3] = 0.3

            ax.imshow(mask_rgba)

            contours = plt.contour(
                seg_mask, levels=[0.5], colors=['orange'], linewidths=2)

            legend_patches.append(
                mpatches.Patch(color=color, label=f"Segment {i+1}"))

            self.logger.debug(f"Processed overlay {i+1} of {len(valid_masks)}")

        ax.legend(handles=legend_patches, loc='upper right')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "output.png"))
        plt.close()

        self.logger.info("Segmentation complete")
        return output_paths

    def segment(
            self, image_rgb, output_dir="output", boxes=None, points=None, labels=None):
        self.logger.info("Starting segmentation process")
        try:
            if points is not None and labels is not None:
                self.logger.info(f"Segmenting from {len(points)} points")
                masks = self.segment_from_points(image_rgb, points, labels)
            elif boxes is not None and len(boxes) > 0:
                self.logger.info(f"Segmenting from {len(boxes)} boxes")
                masks = self.segment_from_boxes(image_rgb, boxes)
            else:
                self.logger.info(f"No boxes, generating masks")
                masks = self.generate_masks(image_rgb)

            if not masks or len(masks) == 0:
                self.logger.error("No masks returned by SAM.")
                raise RuntimeError("No masks returned by SAM.")\

            return self.apply_masks(image_rgb, masks, output_dir)
        except Exception as e:
            self.logger.error(f"Segmentation error: {e}")
            return []

    def segment_from_boxes(self, image_rgb, boxes):
        self.predictor.set_image(image_rgb)
        masks = []

        if isinstance(boxes, np.ndarray):
            boxes_list = boxes
        else:
            boxes_list = np.array(boxes)

        if len(boxes_list.shape) == 1:
            self.logger.info(f"Single box, shape: {boxes_list.shape}")
            boxes_list = boxes_list[np.newaxis, :]
            self.logger.info(
                f"Shape of single box after adding dim: {boxes_list.shape}")

        for box in boxes_list:
            input_box = box[np.newaxis, :]
            masks_pred, scores, logits = self.predictor.predict(
                box=input_box,
                multimask_output=False
            )

            mask = masks_pred[0]

            masks.append({
                "segmentation": mask,
                "area": float(mask.sum()),
                "bbox": box.tolist()
            })
        return masks

    def _get_bbox_from_mask(self, mask):
        y_indices, x_indices = np.where(mask)
        x1, x2 = x_indices.min(), x_indices.max()
        y1, y2 = y_indices.min(), y_indices.max()
        return [x1, y1, x2, y2]

    def segment_from_points(self, image_rgb, points, labels):
        self.predictor.set_image(image_rgb)

        input_point = np.array(points)
        input_label = np.array(labels)

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True  # Let user choose best mask
        )

        results = []
        for i, mask in enumerate(masks):
            results.append({
                "segmentation": mask,
                "area": float(mask.sum()),
                "bbox": self._get_bbox_from_mask(mask)
            })
        return results
