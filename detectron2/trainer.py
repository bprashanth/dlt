import os
import logging
import argparse
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo
from coco import CocoHelper


def setup_logger(log_level=logging.INFO):
    """Configure root logger for the entire application."""
    # Remove any existing handlers
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Setup handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)


class Trainer:
    def __init__(
            self, train_dir, val_dir, output_dir, focus_label=None, logger=None):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = logger or logging.getLogger("Trainer")

        # TODO(prashanth@): raise exception if dir doesn't exist
        self.train_coco_path = os.path.join(self.train_dir, "annotations.json")
        self.val_coco_path = os.path.join(self.val_dir, "annotations.json")

        self.train_coco = CocoHelper(
            self.train_coco_path, focus_label=focus_label)
        self.val_coco = CocoHelper(
            self.val_coco_path, focus_label=focus_label)

    def _register_datasets(self):

        # register_coco_instances is a wrapper around helper functions that
        # translate COCO format to Detectron2 format.
        register_coco_instances(
            "train_dataset", {}, self.train_coco_path, self.train_dir
        )
        register_coco_instances(
            "val_dataset", {}, self.val_coco_path, self.val_dir
        )

        class_names = self.train_coco.get_class_names()

        # Register metadata
        MetadataCatalog.get("train_dataset").set(thing_classes=class_names)
        MetadataCatalog.get("val_dataset").set(thing_classes=class_names)

    def _get_config(self):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        )

        cfg.DATASETS.TRAIN = ("train_dataset",)
        cfg.DATASETS.TEST = ("val_dataset",)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.train_coco.get_num_classes()
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 1000
        cfg.SOLVER.STEPS = []
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cpu"
        cfg.OUTPUT_DIR = self.output_dir
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        return cfg

    def train(self):
        self._register_datasets()
        cfg = self._get_config()
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        checkpointer = DetectionCheckpointer(trainer.model)
        checkpoint_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        checkpointer.save(checkpoint_path)
        return checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description="Launch training inside Docker container")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Path to training directory")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Path to validation directory")
    parser.add_argument("--output_dir", type=str,
                        default="output", help="Path for model output")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--focus_label", type=str, default="Lantana camara",
                        help="Field to focus on for validation")
    args = parser.parse_args()

    setup_logger(getattr(logging, args.log_level.upper()))

    logging.info(f"Starting training with \n\
        train_dir: {args.train_dir}\n\
        val_dir: {args.val_dir}\n\
        output_dir: {args.output_dir}\n\
        log_level: {args.log_level}\n\
    ")

    trainer = Trainer(args.train_dir, args.val_dir,
                      args.output_dir, args.focus_label)
    checkpoint_path = trainer.train()
    logging.info(f"Training completed and model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
