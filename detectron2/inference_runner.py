"""
This script is used to run inference on a single image. 

NB: The classes supplied via the --classes argument are used to filter the predictions. You can find class names in the test/annotations.json file, or by running the following command: 
    python3 ./hack/type_name_combos.py

Usage: 
    python inference_runner.py --weights_path path/to/weights.pth \
        --test_dir path/to/test \
        --output_dir path/to/output \
        --test_image path/to/image.jpg \
        --classes "Lantana Cover,Trees" \
        --minimal_visualization
"""
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
from annotation import CocoHelper
from inference_utils import DetectronVisualizer
import gradio as gr


class GradioLauncher:
    def __init__(self, weights_path, output_dir, test_dir):
        """
        Initializes the Gradio UI wrapper for interactive inference.

        @param weights_path: Path to the trained Detectron2 weights (.pth) file.
        @param output_dir: Directory where any overlay or logs may be optionally saved (also used by underlying visualizer).
        @param test_dir: Directory containing the 'annotations.json' file, used to load the class names expected by the model.

        This class wraps the InferenceRunner and exposes a UI via Gradio.
        The user can upload an image, set the confidence threshold,
        and choose which classes to visualize and export in JSON format.
        """
        self.runner = InferenceRunner(
            weights_path=weights_path,
            output_dir=output_dir,
            test_dir=test_dir,
        )

    def launch(self):
        def predict_fn(image, confidence_thresh, selected_classes):
            self.runner.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_thresh
            self.runner.predictor = DefaultPredictor(self.runner.cfg)

            # Image from Gradio is already in RGB, no need to convert
            outputs = self.runner.predict_image(image)

            # Get visualization (which returns RGB since it uses matplotlib)
            vis = self.runner.get_visualized_image(
                image, outputs)

            # No need for additional conversion since vis is already in RGB
            json_data = self.runner.get_coco_json(
                outputs, "gradio_input.jpg")
            return vis, json_data

        gr.Interface(
            fn=predict_fn,
            inputs=[
                gr.Image(type="numpy", label="Upload Image"),
                gr.Slider(0.1, 1.0, step=0.05, value=0.5,
                          label="Confidence Threshold"),
                gr.CheckboxGroup(self.runner.class_names,
                                 label="Select Classes to Show")
            ],
            outputs=[
                gr.Image(label="Prediction Overlay"),
                gr.JSON(label="COCO Predictions")
            ],
            title="Lantana Detection Inference",
            description="Upload a drone image, adjust threshold and class filters, and see the predictions."
        ).launch()


class InferenceRunner:
    def __init__(self, weights_path, output_dir, test_dir, confidence_threshold=0.3, test_image=None, minimal_visualization=False, selected_classes=None):
        self.weights_path = weights_path
        self.output_dir = output_dir
        self.test_dir = test_dir
        self.confidence_threshold = confidence_threshold
        self.test_image = test_image
        # Only try to get input_basename if test_image is provided
        if test_image is not None:
            self.input_basename = os.path.basename(test_image)
        else:
            self.input_basename = None

        # Get the base name of the input image and add 'inference_' prefix
        # Eg:
        # input: foo.png
        # output image with inference polygon: inference_foo.png
        if self.input_basename:
            output_filename = f"inference_{self.input_basename}"
        else:
            output_filename = "inference_unknown"
        self.output_png = os.path.join(output_dir, output_filename)

        # Create JSON filename based on test image name without extension
        # Eg:
        # input: foo.png
        # output json: predictions_foo.json
        if self.input_basename:
            # Remove the file extension to get the base name
            base_name_without_ext = os.path.splitext(self.input_basename)[0]
            json_filename = f"predictions_{base_name_without_ext}.json"
        else:
            json_filename = "predictions_unknown.json"
        self.output_json = os.path.join(output_dir, json_filename)
        self.logger = logging.getLogger("inference")

        self.test_coco_path = os.path.join(test_dir, "annotations.json")
        self.test_coco = CocoHelper(self.test_coco_path)
        self.class_names = self.test_coco.get_class_names()

        self.cfg = self._get_config()
        self.logger.info(
            f"Loaded {len(self.class_names)} classes from {self.test_coco_path}")

        self.predictor = DefaultPredictor(self.cfg)
        self.visualizer = DetectronVisualizer(
            self.class_names, selected_classes, minimal_visualization)

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
        image = cv2.imread(self.test_image)
        outputs = self.predictor(image)
        self.logger.info(outputs["instances"])

        self.visualizer.draw_predictions(
            image, outputs, self.output_png)
        self.visualizer.save_predictions_json(
            outputs, self.test_image, self.output_json)
        self.logger.info(f"Saved overlay to {self.output_png}")
        self.logger.info(f"Saved COCO-style predictions to {self.output_json}")

    def predict_image(self, image):
        return self.predictor(image)

    def get_visualized_image(self, image, outputs):
        return self.visualizer.get_overlay(image, outputs)

    def get_coco_json(self, outputs, image_path):
        return self.visualizer.format_predictions_as_json(
            outputs, image_path)


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
    parser.add_argument("--gradio_mode", action="store_true",
                        help="Run the Gradio interface instead of the command line interface.")
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated list of class names to filter predictions. If not provided, all classes will be shown.")
    parser.add_argument("--minimal_visualization", action="store_true",
                        help="Use minimal visualization (no legends, borders, or text).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Parse the classes argument if provided
    selected_classes = None
    if args.classes:
        selected_classes = [c.strip() for c in args.classes.split(",")]
        logging.info(f"Filtering predictions for classes: {selected_classes}")

    if args.gradio_mode:
        logging.info("Running in interactive (UI) mode...")
        launcher = GradioLauncher(
            weights_path=args.weights_path,
            output_dir=args.output_dir,
            test_dir=args.test_dir
        )
        launcher.launch()
    else:
        logging.info(
            f"Running inference with weights: {args.weights_path} and confidence threshold: {args.confidence_threshold}")
        runner = InferenceRunner(
            weights_path=args.weights_path,
            output_dir=args.output_dir,
            test_dir=args.test_dir,
            confidence_threshold=args.confidence_threshold,
            test_image=args.test_image,
            minimal_visualization=args.minimal_visualization,
            selected_classes=selected_classes
        )
        # Pass selected_classes to run method
        runner.run()


if __name__ == "__main__":
    main()
