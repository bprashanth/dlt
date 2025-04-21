"""
This script is used to run inference on a single image using a pre-trained Detectron2 model.

Usage: 
python main.py --segmentation_type {instance | panoptic} --image path/to/image.jpg --output_dir path/to/output/dir
"""
import argparse
import logging
import os
import numpy as np
import torch
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_panoptic

COCO_PANOPTIC_CATEGORIES = "detectron2://COCO-PanopticSegmentation/panoptic_coco_categories.json"

CLASSES = ["Cells", "Mitochondria", "Alpha Granules", "Carnicular Vessel"]
DEFAULT_CATALOG_INSTANCE = "coco_2017_val"
DEFAULT_CATALOG_PANOPTIC = "coco_2017_panoptic_val"


def setup_logger(log_level=logging.INFO):
    logger = logging.getLogger("detectron2")
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_config(segmentation_type="instance", score_threshold=0.5, weights_path=None, logger=None):
    """
    Get the configuration for the predictor based on segmentation type.
    """
    cfg = get_cfg()

    if segmentation_type == "instance":
        config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(config_file))

        if weights_path:
            # Custom weights: use our 4 classes
            logger.info(
                f"Using custom weights from {weights_path} for instance segmentation.")
            MetadataCatalog.get("custom_catalog").set(
                thing_classes=CLASSES
            )
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)
            cfg.MODEL.WEIGHTS = weights_path
        else:
            # Model zoo weights: use COCO classes
            logger.info("Using model zoo weights with COCO classes")
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
            # Let Detectron2 use its default COCO metadata

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    else:
        config_file = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
        if weights_path:
            logger.info(
                f"Ignoring weights from {weights_path} and using weights from model zoo: {config_file}, since panoptic segmentation is not supported with custom weights.")
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

    return cfg


def run_inference_instance(image_path, cfg, logger, output_dir, min_area_ratio, device=None):
    """Run instance segmentation inference."""
    logger.info(f"Running instance segmentation inference on {image_path}")

    img = cv2.imread(image_path)
    if device:
        cfg.MODEL.DEVICE = device

    predictor = DefaultPredictor(cfg)
    outputs = predictor(img)
    instances = outputs["instances"]

    # Use "inference" metadata for custom weights, or COCO metadata for model zoo
    if hasattr(cfg.MODEL, "WEIGHTS") and cfg.MODEL.WEIGHTS.endswith("model_final.pth"):
        metadata = MetadataCatalog.get("custom_catalog")
    else:
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    visualizer = Visualizer(img[:, :, ::-1], metadata)
    vis_output = visualizer.draw_instance_predictions(instances.to("cpu"))

    output_image_path = os.path.join(output_dir, "output_image.jpg")
    logger.info(f"Saving visualized image to {output_image_path}")
    cv2.imwrite(output_image_path, vis_output.get_image()[:, :, ::-1])

    masks = instances.pred_masks.to("cpu").numpy()
    for idx, mask in enumerate(masks):
        mask_image_path = os.path.join(output_dir, f"mask_{idx}.png")
        logger.info(f"Saving mask {idx} to {mask_image_path}")
        cv2.imwrite(mask_image_path, (mask * 255).astype(np.uint8))


def register_metadata_panoptic():
    """Register COCO panoptic metadata if not already registered."""
    metadata = MetadataCatalog.get("coco_2017_panoptic_val")
    # Check if metadata is empty by trying to access its dictionary
    if not hasattr(metadata, "stuff_classes"):
        register_coco_panoptic(
            "coco_2017_panoptic_val",
            {},
            "dummy/path",  # We don't need actual data paths for inference
            "dummy/path",
            COCO_PANOPTIC_CATEGORIES
        )


def run_inference_panoptic(image_path, cfg, logger, output_dir, min_area_ratio, device=None):
    """Run panoptic segmentation inference."""
    logger.info(f"Running panoptic segmentation inference on {image_path}")

    img = cv2.imread(image_path)
    if device:
        cfg.MODEL.DEVICE = device

    predictor = DefaultPredictor(cfg)
    outputs = predictor(img)
    panoptic_seg, segments_info = outputs["panoptic_seg"]

    visualizer = Visualizer(
        img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    vis_output = visualizer.draw_panoptic_seg_predictions(
        panoptic_seg.to("cpu"), segments_info)

    output_image_path = os.path.join(output_dir, "output_image.jpg")
    logger.info(f"Saving visualized image to {output_image_path}")
    cv2.imwrite(output_image_path, vis_output.get_image()[:, :, ::-1])

    # Save the panoptic segmentation mask
    mask_image_path = os.path.join(output_dir, "panoptic_mask.png")
    logger.info(f"Saving panoptic mask to {mask_image_path}")
    cv2.imwrite(mask_image_path, panoptic_seg.to(
        "cpu").numpy().astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="Detectron2 Object Detection")
    parser.add_argument("--log_level", type=str, default="INFO", choices=[
                        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Log level")
    parser.add_argument("--segmentation_type", type=str, default="instance",
                        choices=["instance", "panoptic"], help="Type of segmentation to perform - this chooses the model weights and configs.")
    parser.add_argument("--score_threshold", type=float,
                        default=0.5, help="Score threshold, recommended to be 0.5 - 0.8")
    parser.add_argument("--image", type=str, default="",
                        required=True, help="Image path")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory to save masks and results")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cuda", "cpu"], help="Device to use for inference")
    parser.add_argument("--min_area_ratio", type=float, default=0.01,
                        help="Minimum area ratio of the detected object")
    parser.add_argument("--weights_path", type=str,
                        default=None,
                        help="Path to the weights file to use for inference. Must correspond to the model architecture, i.e --segmentation_type instance needs maskrcnn weights and panoptic segmentation corresponds to weights of panoptic fpn.")

    args = parser.parse_args()

    logger = setup_logger(args.log_level)

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = get_config(
        segmentation_type=args.segmentation_type,
        score_threshold=args.score_threshold,
        weights_path=args.weights_path,
        logger=logger
    )

    if args.segmentation_type == "instance":
        run_inference_instance(args.image, cfg, logger, args.output_dir,
                               args.min_area_ratio, args.device)
    else:
        run_inference_panoptic(args.image, cfg, logger, args.output_dir,
                               args.min_area_ratio, args.device)


if __name__ == "__main__":
    main()
