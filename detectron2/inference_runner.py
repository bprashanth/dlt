import os
import cv2
import json
import logging
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from coco import CocoHelper
from inference_utils import DetectronVisualizer


class InferenceRunner:
    def __init__(self, weights_path, test_image, output_dir, test_dir, confidence_threshold):
        self.weights_path = weights_path
        self.test_image = test_image
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold

        # Get the base name of the input image and add 'inference_' prefix
        input_basename = os.path.basename(test_image)
        output_filename = f"inference_{input_basename}"
        self.output_png = os.path.join(output_dir, output_filename)

        self.output_json = os.path.join(output_dir, "predictions.json")
        self.logger = logging.getLogger("inference")

        self.test_coco_path = os.path.join(test_dir, "annotations.json")
        self.test_coco = CocoHelper(self.test_coco_path)
        self.class_names = self.test_coco.get_class_names()

        self.cfg = self._get_config()
        self.logger.info(
            f"Loaded {len(self.class_names)} classes from {self.test_coco_path}")

        self.visualizer = DetectronVisualizer(output_dir, self.class_names)

    def _get_config(self):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        )
        cfg.MODEL.WEIGHTS = self.weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.class_names)
        cfg.MODEL.DEVICE = "cpu"
        return cfg

    def run(self):
        MetadataCatalog.get("inference_dataset").set(
            thing_classes=self.class_names)
        predictor = DefaultPredictor(self.cfg)
        image = cv2.imread(self.test_image)
        outputs = predictor(image)
        self.logger.info(outputs["instances"])

        self.visualizer.draw_predictions(image, outputs, self.output_png)
        self.visualizer.save_predictions_json(
            outputs, self.test_image, self.output_json)
        self.logger.info(f"Saved overlay to {self.output_png}")
        self.logger.info(f"Saved COCO-style predictions to {self.output_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, required=True,
                        help="Path to the trained model weights.")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to the directory containing the test annotations.json - the classes will be extracted from this file. This does NOT have to include the test_image, it just needs the same structure as the train_dir.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to output directory where the overlay and predictions will be saved.")
    parser.add_argument("--test_image", type=str, required=True, default=None,
                        help="Path to test image on which we predict classes.")
    parser.add_argument("--confidence_threshold", type=float, default=0.3,
                        help="Confidence score threshold for predictions (default: 0.3)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    logging.info(
        f"Running inference with weights: {args.weights_path} and confidence threshold: {args.confidence_threshold}")
    runner = InferenceRunner(
        weights_path=args.weights_path,
        test_image=args.test_image,
        output_dir=args.output_dir,
        test_dir=args.test_dir,
        confidence_threshold=args.confidence_threshold
    )
    runner.run()


if __name__ == "__main__":
    main()
