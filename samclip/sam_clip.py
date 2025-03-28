"""
SAM Segmenter with CLIP

This script uses SAM to segment an image. Invocation: 

$ cd samclip
$ docker run -it --rm \
  -v $(pwd):/app \
  -w /app \
  sam-clip \
  bash
# rm -rf ./output/* && python sam_clip.py --image drone.jpg --output_dir ./output/

TODO:
- Add CLIP to generate captions for each segment
- There is some discrepancy between min_area_ratio as passed to SAM and the code
  that applies the masks. Both should be the same, but the former generates 33 masks and the latter discards 29 of those. 
"""
import torch
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import random
import numpy as np
import logging
from tqdm import tqdm
import clip
from PIL import Image


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

    def segment(self, image_rgb, output_dir="output"):
        self.logger.info("Starting segmentation process")
        try:
            masks = self.generate_masks(image_rgb)
            if len(masks) == 0:
                self.logger.error("No masks returned by SAM.")
                raise RuntimeError("No masks returned by SAM.")
            return self.apply_masks(image_rgb, masks, output_dir)
        except Exception as e:
            self.logger.error(f"Segmentation error: {e}")
            return []


class CLIPScorer:

    def __init__(self, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.logger = logging.getLogger('CLIPScorer')
        self.logger.info(f"Initialized CLIP scorer on device: {self.device}")

    def score(self, image_path, prompt):
        self.logger.info(f"Scoring image: {image_path}")
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            text_input = clip.tokenize([prompt]).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)
                similarity = torch.nn.functional.cosine_similarity(
                    image_features, text_features)

            return similarity.item()
        except Exception as e:
            self.logger.error(f"Error scoring image: {e}")
            return 0.0


def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('SAMSegmenter')


def get_color():
    """Generate a bright, high-contrast color using HSV color space."""
    hue = random.random()  # Random hue (0-1)
    saturation = 0.8 + random.random() * 0.2  # High saturation (0.8-1.0)
    value = 0.8 + random.random() * 0.2  # High brightness (0.8-1.0)

    # Convert HSV to RGB (assuming values in range 0-1)
    h = hue * 6
    i = int(h)
    f = h - i
    p = value * (1 - saturation)
    q = value * (1 - f * saturation)
    t = value * (1 - (1 - f) * saturation)

    if i == 0:
        return [value, t, p]
    elif i == 1:
        return [q, value, p]
    elif i == 2:
        return [p, value, t]
    elif i == 3:
        return [p, q, value]
    elif i == 4:
        return [t, p, value]
    else:
        return [value, p, q]


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM segmentation on an image")

    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set the logging level")

    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save the output masks")
    parser.add_argument("--model_type", type=str,
                        default="vit_b", help="Type of SAM model to use")
    parser.add_argument("--checkpoint_path", type=str,
                        default="segment-anything/sam_vit_b_01ec64.pth", help="Path to the SAM checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for segmentation")
    parser.add_argument("--min_area_ratio", type=float, default=0.01,
                        help="Minimum area ratio of the segment to be included")

    args = parser.parse_args()

    logger = setup_logging(getattr(logging, args.log_level))
    logger.info("Initialized logging...")

    image = cv2.imread(args.image)
    if image is None:
        logger.error(f"Could not load image: {args.input}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logger.info(f"Loaded image with shape: {image_rgb.shape}")

    segmenter = SAMSegmenter(
        args.model_type, args.checkpoint_path, args.device)
    output_paths = segmenter.segment(image_rgb, output_dir=args.output_dir)

    logger.info(f"Segmentation complete. Masks saved to: {output_paths}")

    clip_scorer = CLIPScorer()

    prompt = "a dense patch of lantana camara seen from a drone, typical of the Nilgiris mountain range, with tangled dark green foliage forming bushy, uneven clusters. The patch may or may not have flowers, and often appears in contrast to the surrounding grass or open ground."

    logger.info(f"Scored {len(output_paths)} segments with CLIP")
    for path in output_paths:
        score = clip_scorer.score(path, prompt)
        if score is not None:
            logger.info(f"{os.path.basename(path)}: {score:.4f}")


if __name__ == "__main__":
    main()
