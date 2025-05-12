# isort: skip_file
# fmt: off
# ruff: noqa: E402
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_utils import DetectronVisualizer
from coco import CocoHelper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay COCO annotations on a PNG image")
    parser.add_argument("--png", required=True, help="Path to the PNG image")
    parser.add_argument("--coco", required=True,
                        help="Path to the COCO annotations JSON file")
    parser.add_argument("--output", default="coco_overlay.png",
                        help="Path to save the output image")
    args = parser.parse_args()

    test_coco = CocoHelper(args.coco)
    class_names = test_coco.get_class_names()

    visualizer = DetectronVisualizer(class_names)
    visualizer.draw_coco_annotations(args.png, args.coco, args.output)
    print(f"Saved COCO PNG to {args.output}")
