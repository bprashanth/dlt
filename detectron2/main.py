"""Command line interface for dataset validation."""

import argparse
import logging
import json
from validation import DataPreprocessor


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


def main():
    parser = argparse.ArgumentParser(
        description="Validate new datasets before copying them into app/data")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Path to the root directory containing site directories")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["INFO", "DEBUG",
                                 "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")

    args = parser.parse_args()

    # Setup logging at application start
    setup_logger(getattr(logging, args.log_level.upper()))

    # Initialize the preprocessor
    preprocessor = DataPreprocessor(args.root_dir)

    # Run validation
    results = preprocessor.validate()
    print("\nValidation Results:")
    print(json.dumps(results, indent=2))

    # Get class information
    classes = preprocessor.get_classes()
    print("\nClass Distribution:")
    print(classes)


if __name__ == "__main__":
    main()
