"""Validate COCO annotation file.

This script uses the same class the trainer uses to validate COCO.
"""
# Add the parent directory to Python path to import trainer

# isort: skip_file
# fmt: off
# ruff: noqa: E402
import os
import sys
import logging
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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



def validate_coco_file(coco_path, focus_label=None):
    try:
        # This will automatically validate the file when loading
        coco_helper = CocoHelper(coco_path, focus_label=focus_label)

        # Print some statistics
        logging.info(f"Validation successful!")
        logging.info(f"Number of classes: {coco_helper.get_num_classes()}")
        logging.info(f"Class names: {coco_helper.get_class_names()}")
        return True
    except ValueError as e:
        logging.error(f"Validation failed: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during validation: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate COCO annotation file")
    parser.add_argument("--coco", type=str, required=True,
                        help="Path to COCO annotation JSON file")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--focus_field", type=str, default="Lantana camara",
                        help="Field to focus on for validation")
    args = parser.parse_args()

    setup_logger(getattr(logging, args.log_level.upper()))

    logging.info(f"Validating COCO file: {args.coco}")
    success = validate_coco_file(args.coco, args.focus_field)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
