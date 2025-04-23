import argparse
import logging
import os
import json
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
import cv2
import numpy as np
from detectron2.structures import BoxMode

# TODO(prashanth@): make this configurable.
CLASSES = ["Cells", "Mitochondria", "Alpha Granule", "Carnicular Vessel"]


def setup_logger(log_level=logging.INFO):
    logger = logging.getLogger("detectron2")
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_config(config_file="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
               model_weights=None,
               num_classes=None,
               score_threshold=0.5,
               training=False):
    """
    Get Detectron2 config.

    @param config_file: Path to model config yaml
    @param model_weights: Path to custom weights file. If None, use model_zoo weights
    @param num_classes: Number of classes. Only set when using custom weights
    @param score_threshold: Detection confidence threshold
    @param training: Whether this is for training (True) or inference (False)
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))

    if model_weights:
        # Using custom weights - set num_classes if provided
        cfg.MODEL.WEIGHTS = model_weights
        if num_classes is not None:
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    else:
        # Using model_zoo weights
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        if training:
            # Training mode - set num_classes even with model_zoo weights
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold

    # CPU-friendly configuration: how many cpus to use for data loading?
    # 0 means use the main process and let it take available cpus.
    # We only have 4 images, they will all fit in memory.
    cfg.DATALOADER.NUM_WORKERS = 2

    # Batch size (number of images processed in parallel)
    # Larger batch needs more memory
    cfg.SOLVER.IMS_PER_BATCH = 2

    # Number of training iterations
    # Typically set to (#images * #epochs)/batch size
    #
    # Epoch1:  Iterations on [batch size]
    # Iteration 1: images [1, 2]
    # Iteration 2: images [3, 4]
    #
    # Epoch2:
    # Iteration 3: images [1, 2]
    # Iteration 4: images [3, 4]
    # ...
    #
    # We will see the entire dataset every 2 iterations and over 100 iterations
    # we will see the entire dataset 50 times, over a total of 100 iterations.
    cfg.SOLVER.MAX_ITER = 1000  # (4 * 50) // 2

    # Learning rate: what is the step size for adjusting parameters when a
    # mistake is made?
    # Smaller: slower convergence, more accurate
    # Larger: faster convergence, less accurate
    cfg.SOLVER.BASE_LR = 0.00025

    # Learning rate decay (don't decay)
    # With larger datasets, we want the learning rate to decay with iterations.
    # We want to make smaller adjustments as we get closer to the answer.
    # With just 4 images, we don't need decay.
    cfg.SOLVER.STEPS = []
    # Batch size per image, default is 512
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256

    cfg.MODEL.DEVICE = "cpu"
    return cfg


def register_dataset(data_path, dataset_name, classes=CLASSES, logger=None):
    """
    Register a dataset with Detectron2.

    @param data_path: Path to the data directory containing annotations.json and images/
    @param dataset_name: Name to register the dataset as (e.g., "train_dataset" or "val_dataset")
    @param classes: List of class names
    @param logger: Logger instance
    """

    if dataset_name in DatasetCatalog:
        logger.info(
            f"Dataset '{dataset_name}' is already registered. Unregistering...")
        DatasetCatalog.remove(dataset_name)

    logger.info(f"Registering dataset {dataset_name}...")
    annotation_file = os.path.join(data_path, "annotations.json")
    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    # Create category ID mapping (subtract 1 since annotations use 1-based indexing)
    cat_id_to_index = {cat['id']: cat['id'] -
                       1 for cat in annotations.get('categories', [])}

    def get_dataset_dicts():
        dataset_dicts = []
        for img in annotations['images']:
            record = {}
            record["file_name"] = os.path.join(
                data_path, img['file_name'])
            record["height"] = img['height']
            record["width"] = img['width']
            record["image_id"] = img['id']

            # Find annotations for this image
            anns_for_image = [
                ann for ann in annotations['annotations']
                if ann['image_id'] == img['id']
            ]

            objs = []
            for ann in anns_for_image:
                # Map 1-based category_id to 0-based index
                category_idx = cat_id_to_index[ann['category_id']]
                obj = {
                    "bbox": ann['bbox'],
                    "bbox_mode": BoxMode.XYWH_ABS,  # COCO format uses XYWH_ABS
                    "segmentation": ann['segmentation'],
                    "category_id": category_idx,  # This will now be 0-based
                    "iscrowd": ann.get('iscrowd', 0)
                }
                objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

            # Debug logging for first record
            if len(dataset_dicts) == 1:
                logger.info("First record structure:")
                logger.info(json.dumps(
                    dataset_dicts[0], indent=2, default=str))

        return dataset_dicts

    DatasetCatalog.register(dataset_name, get_dataset_dicts)
    MetadataCatalog.get(dataset_name).set(thing_classes=classes)


def register_metadata(dataset_name, classes=CLASSES, logger=None):
    """
    Register only the metadata (class names) without actual dataset.
    Used for inference when we don't have ground truth annotations. 
    """
    if dataset_name in MetadataCatalog:
        MetadataCatalog.remove(dataset_name)
    MetadataCatalog.get(dataset_name).set(thing_classes=classes)


def train_model(cfg, output_dir, logger=None):
    logger.info("Starting training...")
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.TEST = ("val_dataset",)  # Add validation dataset
    cfg.TEST.EVAL_PERIOD = 100  # Evaluate every 100 iterations
    cfg.OUTPUT_DIR = os.path.join(output_dir, "output")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)

    # Start from scratch every training run, but load the COCO weights from
    trainer.resume_or_load(resume=False)
    trainer.train()

    checkpointer = DetectionCheckpointer(trainer.model)
    checkpointer.save(os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
    logger.info("Training completed successfully.")


def run_inference(cfg, weights_path, inference_data, output_dir, logger=None):
    """
    Run inference on a given dataset using a trained model.

    @param cfg: Detectron2 configuration object
    @param weights_path: Path to the trained model weights
    @param inference_data: Path to the directory containing images for inference
    @param output_dir: Path to the output directory for storing inference results
    @param logger: Logger instance
    """
    logger.info("Starting inference...")

    if weights_path and os.path.exists(weights_path):
        logger.info(f"Loading custom weights from {weights_path}")
        cfg.MODEL.WEIGHTS = weights_path
        metadata = MetadataCatalog.get("train_dataset")
    else:
        logger.info("Using default COCO weights and metadata")
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        register_metadata(
            cfg.DATASETS.TRAIN[0],
            classes=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
        )

    predictor = DefaultPredictor(cfg)
    output_images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]
    image_files = [
        f for f in os.listdir(inference_data)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]

    for image_file in image_files:
        image_path = os.path.join(inference_data, image_file)
        logger.info(f"Processing {image_path}")

        # Reead image
        im = cv2.imread(image_path)

        outputs = predictor(im)

        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=1.0)
        result = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        output_path = os.path.join(output_images_dir, f"pred_{image_file}")
        cv2.imwrite(output_path, result.get_image()[:, :, ::-1])

    logger.info(f"Inference completed. Results saved in {output_images_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train and run inference for Detectron2.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=[
                        "INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    parser.add_argument("--train_data",  type=str,
                        required=False, default='./data/train', help="Path to the training data")
    parser.add_argument("--val_data", type=str,
                        required=False, default="./data/val", help="Path to the validation data")
    parser.add_argument("--train_config", type=str,
                        required=False, help="Path to the training config file")
    parser.add_argument("--inference_data", type=str,
                        default=None, help="Path to the inference data")
    parser.add_argument("--output_dir", type=str,
                        default="output", help="Path to the output directory. This is where post inference images, checkpoints, etc. are stored.")
    parser.add_argument("--weights_path", type=str,
                        required=False, default=None, help="Path to the trained model weights")

    # TODO(prashanth@): add more arguments, eg number of classes, batch size,
    # learning rate, etc.

    args = parser.parse_args()
    logger = setup_logger(args.log_level)

    inference_dir = os.path.join(args.output_dir, "inference")
    os.makedirs(inference_dir, exist_ok=True)

    if args.train_data:
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Register both training and validation datasets
        register_dataset(args.train_data, "train_dataset",
                         classes=CLASSES, logger=logger)
        register_dataset(args.val_data, "val_dataset",
                         classes=CLASSES, logger=logger)
        cfg = get_config(
            model_weights=None,
            num_classes=len(CLASSES),
            training=True  # Indicate this is for training
        )
        train_model(cfg, checkpoint_dir, logger=logger)
    else:
        # Inference mode
        if args.weights_path:
            register_metadata("train_dataset", classes=CLASSES)
            cfg = get_config(
                model_weights=args.weights_path,
                num_classes=len(CLASSES),
                training=False
            )
        else:
            cfg = get_config(training=False)

        run_inference(cfg, args.weights_path, args.inference_data,
                      args.output_dir, logger=logger)


if __name__ == "__main__":
    main()
