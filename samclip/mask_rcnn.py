import torch
import torchvision
from torchvision.transforms import functional as F
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import logging
from tqdm import tqdm
from utils import get_color


class MaskRCNNSegmenter:
    def __init__(self, device=None, min_area_ratio=0.01, score_threshold=0.5):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.min_area_ratio = min_area_ratio
        self.score_threshold = score_threshold

        self.logger = logging.getLogger('MaskRCNNSegmenter')
        self.logger.info(
            f"Initialized MaskRCNN on device: {self.device}. Using min_area_ratio: {self.min_area_ratio} and score_threshold: {self.score_threshold}.")

        try:
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                pretrained=True)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            self.logger.error(f"Error loading MaskRCNN model: {str(e)}")
            raise

    def generate_masks(self, image_rgb):
        image_tensor = F.to_tensor(image_rgb).to(self.device)

        with torch.no_grad():
            predictions = self.model([image_tensor])[0]

        masks = []
        h, w = image_rgb.shape[:2]
        min_area = self.min_area_ratio * h * w

        for i in range(len(predictions['scores'])):
            score = predictions['scores'][i].item()
            label = predictions['labels'][i].item()
            mask = predictions['masks'][i, 0].cpu().numpy() > 0.5
            area = mask.sum()

            if score >= self.score_threshold and area >= min_area:
                x1, y1, x2, y2 = predictions['boxes'][i].cpu(
                ).numpy().astype(int)
                bbox = [x1, y1, x2, y2]

                masks.append({
                    'segmentation': mask,
                    'score': score,
                    'label': label,
                    'area': area,
                    'bbox': bbox
                })

            return masks

    def apply_masks(
            self, image_rgb, masks, output_dir="output", min_area_ratio=0.01):
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []

        for i, m in tqdm(enumerate(masks), total=len(masks), desc="Processing Mask R-CNN segments"):
            seg_mask = m['segmentation']
            masked_image = image_rgb.copy()
            masked_image[~seg_mask] = [255, 255, 255]
            filename = os.path.join(output_dir, f"maskrcnn_segment_{i+1}.png")

            x1, y1, x2, y2 = m['bbox']
            crop = masked_image[y1:y2, x1:x2]
            plt.imsave(filename, crop)
            output_paths.append(filename)

        # Overlay all segments with labels and scores
        overlay = image_rgb.copy()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(overlay)
        legend_patches = []

        for i, m in enumerate(masks):
            color = get_color()
            seg_mask = m['segmentation']
            rgba_mask = np.zeros((*seg_mask.shape, 4))
            rgba_mask[seg_mask] = list(color) + [0.4]
            ax.imshow(rgba_mask)

            x1, y1, x2, y2 = m['bbox']
            ax.text(x1, y1, f"{i+1}: {m['score']:.2f}", color='white',
                    fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
            legend_patches.append(mpatches.Patch(
                color=color, label=f"Segment {i+1}"))

        ax.legend(handles=legend_patches, loc='upper right')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "maskrcnn_output.png"))
        plt.close()

        return output_paths

    def segment(self, image_rgb, output_dir="output"):
        masks = self.generate_masks(image_rgb)
        if not masks:
            self.logger.error("No masks returned by Mask R-CNN.")
            return []
        return self.apply_masks(image_rgb, masks, output_dir)
