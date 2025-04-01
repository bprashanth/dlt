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
import cv2
import os
import argparse
import logging
from tqdm import tqdm
from grounding_dino import GroundingDINOBoxGenerator
from mask_rcnn import MaskRCNNSegmenter
from sam_segmenter import SAMSegmenter
from clip_scorer import CLIPScorer


def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('SAMSegmenter')


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
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for segmentation")
    parser.add_argument("--min_area_ratio", type=float, default=0.01,
                        help="Minimum area ratio of the segment to be included")
    parser.add_argument("--model", default="samclip",
                        choices=["samclip", "maskrcnn", "groundingdino"], help="Segmentation model to use")

    parser.add_argument("--dino-text_prompt", type=str,
                        default="a clump of brown dots",
                        help="Text prompt for groundingdino and clip")

    parser.add_argument("--clip-text_prompt", type=str,
                        default="drone image of lantan camara.",
                        help="Text prompt for clip")

    parser.add_argument("--dino_config", type=str,
                        default="/models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                        help="Path to GroundingDINO config")

    parser.add_argument("--dino_weights", type=str,
                        default="/models/groundingdino_swint_ogc.pth",
                        help="Path to GroundingDINO weights")

    parser.add_argument("--sam_checkpoint", type=str,
                        default="/models/sam_vit_b_01ec64.pth",
                        help="Path to the SAM checkpoint")

    args = parser.parse_args()

    logger = setup_logging(getattr(logging, args.log_level))
    logger.info("Initialized logging...")

    image = cv2.imread(args.image)
    if image is None:
        logger.error(f"Could not load image: {args.input}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logger.info(f"Loaded image with shape: {image_rgb.shape}")

    if args.model == "maskrcnn":
        segmenter = MaskRCNNSegmenter(min_area_ratio=args.min_area_ratio)
        output_paths = segmenter.segment(image_rgb, output_dir=args.output_dir)
        logger.info(
            f"MaskRCNN: Segmentation complete. Masks saved to: {output_paths}")
        return

    boxes = None
    if args.model == "groundingdino":
        box_generator = GroundingDINOBoxGenerator(
            args.dino_config, args.dino_weights)
        boxes, _, _ = box_generator.generate_boxes(
            image_rgb, args.dino_text_prompt, output_dir=args.output_dir)

    segmenter = SAMSegmenter(
        args.model_type,
        args.sam_checkpoint,
        args.device)

    output_paths = segmenter.segment(
        image_rgb,
        output_dir=args.output_dir,
        boxes=boxes
        # points=[[537, 460]],
        # labels=[1]
    )

    logger.info(f"Segmentation complete. Masks saved to: {output_paths}")

    clip_scorer = CLIPScorer()

    logger.info(f"Scored {len(output_paths)} segments with CLIP")
    for path in output_paths:
        score = clip_scorer.score(path, args.clip_text_prompt)
        if score is not None:
            logger.info(f"{os.path.basename(path)}: {score:.4f}")


if __name__ == "__main__":
    main()
