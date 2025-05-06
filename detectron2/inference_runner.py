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


class InferenceRunner:
    def __init__(self, weights_path, test_image, output_dir, test_dir):
        self.weights_path = weights_path
        self.test_image = test_image
        self.output_dir = output_dir

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

    def _get_config(self):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        )
        cfg.MODEL.WEIGHTS = self.weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
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

        self._draw_predictions(image, outputs)
        self._save_predictions_json(outputs)

    def _draw_predictions(self, image, outputs):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictions = outputs["instances"].to("cpu")

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image_rgb)
        ax.set_title("Predicted Masks")

        categories_present = set()
        cmap = plt.colormaps["tab10"]

        for i in range(len(predictions)):
            mask = predictions.pred_masks[i].numpy()
            class_id = int(predictions.pred_classes[i])
            score = float(predictions.scores[i])
            color = cmap(class_id)
            categories_present.add((class_id, score))

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
                    edgecolor='lime',
                    facecolor=color,
                    alpha=0.2,
                    linewidth=3.0
                )
                ax.add_patch(patch)

        handles = [
            patches.Patch(
                color=cmap(cid),
                label=f"{self.class_names[cid]} ({score:.2f})"
            )
            for cid, score in sorted(categories_present)
        ]
        ax.legend(handles=handles)
        ax.axis('off')
        plt.tight_layout()
        os.makedirs(self.output_dir, exist_ok=True)
        plt.savefig(self.output_png, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        self.logger.info(f"Saved overlay to {self.output_png}")

    def _save_predictions_json(self, outputs):
        predictions = outputs["instances"].to("cpu")
        results = []

        for i in range(len(predictions)):
            mask = predictions.pred_masks[i].numpy()
            class_id = int(predictions.pred_classes[i])
            score = float(predictions.scores[i])

            contours, _ = cv2.findContours(mask.astype(
                np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentations = []

            for contour in contours:
                flattened = contour[:, 0, :].flatten().tolist()
                if len(flattened) >= 6:
                    segmentations.append(flattened)

            result = {
                "image_id": os.path.basename(self.test_image),
                "category_id": class_id,
                "score": score,
                "segmentation": segmentations
            }
            results.append(result)

        with open(self.output_json, "w") as f:
            json.dump(results, f, indent=2)
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    runner = InferenceRunner(
        weights_path=args.weights_path,
        test_image=args.test_image,
        output_dir=args.output_dir,
        test_dir=args.test_dir
    )
    runner.run()


if __name__ == "__main__":
    main()
