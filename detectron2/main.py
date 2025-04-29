"""Command line interface for dataset validation."""

import argparse
import logging
import json
from validation import DataValidator
from discovery import DataDiscovery


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
    # parser.add_argument("--tile_output_dir", type=str, required=True,
    #                     help="Where to save tiles and intermediate JSON")
    # parser.add_argument("--tile_size", type=int,
    #                     default=512, help="Tile size in pixels")
    # parser.add_argument("--overlap", type=int, default=128,
    #                     help="Overlap between tiles in pixels")
    # parser.add_argument("--train_dir", type=str, required=True,
    #                     help="Path to write train images and annotation")
    # parser.add_argument("--val_dir", type=str, required=True,
    #                     help="Path to write val images and annotation")
    # parser.add_argument("--val_split", type=float, default=0.2,
    #                     help="Fraction for validation split")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["INFO", "DEBUG",
                                 "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")

    args = parser.parse_args()

    # Setup logging at application start
    setup_logger(getattr(logging, args.log_level.upper()))

    # Discovery
    discovery = DataDiscovery(args.root_dir)
    discovery_results = discovery.get_discovery_results()

    # Validation
    validator = DataValidator(discovery_results)
    validation_results = validator.validate()
    print("\nValidation Results:")
    print(json.dumps(validation_results, indent=2))

    print("\nClass Distribution:")
    print(validator.get_classes())


if __name__ == "__main__":
    main()
