"""Command line interface for dataset validation."""

import argparse
import logging
import json
import docker
import os
from validation import DataValidator
from discovery import DataDiscovery
from tiling import TileGenerator
from annotation import AnnotationBuilder


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


def launch_training_in_docker(train_dir, val_dir, output_dir, training_image, focus_label, log_level):
    client = docker.from_env()

    # TODO(prashanth@): There is a big assumption here around the working
    # directory being /app.
    container = client.containers.run(
        image=training_image,
        command=[
            "python",
            "/app/trainer.py",
            "--train_dir", train_dir,
            "--val_dir", val_dir,
            "--output_dir", output_dir,
            "--focus_label", focus_label,
            "--resume",
            "--log_level", log_level
        ],
        # volumes={
        #    os.path.abspath("."): {"bind": "/app", "mode": "rw"}
        # },
        working_dir="/app",
        network_mode="host",
        detach=True,
        stdout=True,
        stderr=True,
        remove=True
    )

    logging.info(f"Training container launched with ID: {container.id}")
    for line in container.logs(stream=True):
        # We break log handling here. The container is logging using a logger
        # and the log level is passed down, but if custom handling of logs is
        # setup in this program print might by pass that.
        print(line.decode().strip())
    logging.info("Training completed")


def launch_inference_in_docker(weights_path, test_image, output_dir, image_name, test_dir, confidence_threshold, gradio_mode):
    client = docker.from_env()

    cmd = [
        "python", "inference_runner.py",
        "--weights_path", weights_path,
        "--test_image", test_image,
        "--output_dir", output_dir,
        "--test_dir", test_dir,
        "--confidence_threshold", str(confidence_threshold),
    ]

    if gradio_mode:
        cmd.append("--gradio_mode")

    container = client.containers.run(
        image=image_name,
        command=cmd,
        volumes={os.getcwd(): {"bind": "/app", "mode": "rw"}},
        network_mode="host",
        working_dir="/app",
        detach=True,
        stdout=True,
        stderr=True,
        remove=True
    )

    logging.info(f"Training container launched with ID: {container.id}")
    for line in container.logs(stream=True):
        print(line.decode().strip())
    logging.info("Inference completed")


def main():
    parser = argparse.ArgumentParser(
        description="Validate new datasets before copying them into app/data")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Path to the root directory containing site directories")
    parser.add_argument("--strict", action="store_true", default=False,
                        help="Strict mode - raise an error if any validation fails")
    parser.add_argument("--class_name_key", type=str, default="Name",
                        help="The key in the shapefile that contains the class names - these are the english names of the classes")
    parser.add_argument("--class_id_key", type=str, default="Types",
                        help="The key in the shapefile that contains the class ids - these are the ids of the classes")
    parser.add_argument("--tile_output_dir", type=str, required=True,
                        help="Where to save tiles and intermediate JSON")
    parser.add_argument("--tile_size", type=int,
                        default=512, help="Tile size in pixels")
    parser.add_argument("--overlap", type=float, default=25.0,
                        help="Overlap between tiles in percentage (default 25%)")
    parser.add_argument("--skip_threshold", type=float, default=10.0,
                        help="Minimum percent of valid pixels required in a tile to be included in the dataset. This argument is used to skip tiles that are mostly transparent/nodata. Setting to 10.0 means that tiles with 10% or more valid pixels will be included.")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Path to write train images and annotation")
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Path to write val images and annotation")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction for validation split")
    parser.add_argument("--test_dir", type=str, help="Path to test image dir")
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split")
    parser.add_argument("--focus_label", type=str, default=None,
                        help="Only process annotations for this specific label, eg 'Lantana Cover'")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["INFO", "DEBUG",
                                 "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--no_tile", action="store_true", default=False,
                        help="If True, preserves each TIFF as a single PNG without tiling")
    parser.add_argument("--training_image", type=str, default="detectron2:1.0",
                        help="Docker image to use for training")
    parser.add_argument("--checkpoint_output_dir", type=str, required=True,
                        help="Where to save checkpoints")
    parser.add_argument("--inference_image", type=str, default="detectron2:1.0",
                        help="Docker image to use for inference")
    parser.add_argument("--inference_weights_path", type=str,
                        help="Path to the weights file to use for inference")
    parser.add_argument("--inference_test_image", type=str,
                        help="Path to the test image to use for inference")
    parser.add_argument("--inference_output_dir", type=str,
                        help="Path to the output directory for inference")
    parser.add_argument("--inference_confidence_threshold",
                        type=float, default=0.3,
                        help="Confidence score threshold for predictions (default: 0.3)")
    parser.add_argument("--inference_gradio_mode", action="store_true",
                        default=False,
                        help="Run the Gradio interface instead of the command line interface.")
    parser.add_argument("--pipeline_config", type=str,
                        help="Path to JSON config file controlling pipeline stages")

    args = parser.parse_args()

    # Setup logging at application start
    setup_logger(getattr(logging, args.log_level.upper()))

    # Load pipeline configuration
    pipeline_config = {}
    if args.pipeline_config:
        with open(args.pipeline_config, 'r') as f:
            pipeline_config = json.load(f)

    # Discovery (always runs)
    discovery = DataDiscovery(
        args.root_dir, args.class_name_key, args.class_id_key)
    discovery_results = discovery.get_discovery_results()
    logging.info("Step 1: Discovery")
    logging.debug(json.dumps(discovery_results, indent=2))

    # Validation
    if not pipeline_config.get('skip_validation', False):
        validator = DataValidator(discovery_results, strict=args.strict)
        validation_results = validator.validate()
        logging.debug("\nValidation Results:")
        logging.debug(json.dumps(validation_results, indent=2))
        logging.debug("\nClass Distribution:")
        logging.debug(validator.get_classes())
        logging.info("Step 2: Validation")
    else:
        logging.info("Step 2: Validation (Skipped)")

    # Tile Generation
    if not pipeline_config.get('skip_tiling', False):
        logging.info(f"Step 3: Tile Generation")
        # Calculate overlap in pixels from percentage
        overlap_pixels = int((args.overlap / 100.0) * args.tile_size)
        tile_generator = TileGenerator(
            discovery_results,
            args.tile_output_dir,
            args.tile_size,
            overlap_pixels,
            args.skip_threshold,
            args.no_tile
        )
        tile_metadata_path = tile_generator.generate_tiles()
        logging.debug(f"Tile metadata saved to: {tile_metadata_path}")
    else:
        logging.info("Step 3: Tile Generation (Skipped)")
        # If tiling is skipped, assume tile metadata exists
        tile_metadata_path = f"{args.tile_output_dir}/tiles_metadata.json"

    # Annotation Building
    if not pipeline_config.get('skip_annotation', False):
        logging.info(f"Step 4: Annotation Builder")
        builder = AnnotationBuilder(
            discovery_results,
            args.tile_output_dir,
            tile_metadata_path,
            args.train_dir,
            args.val_dir,
            args.test_dir,
            args.val_split,
            args.test_split,
            seed=args.seed,
            focus_label=args.focus_label,
            no_tile=args.no_tile
        )
        builder.run()
        logging.info("Image and annotations written to:")
        logging.info(f"Train: {args.train_dir}")
        logging.info(f"Val: {args.val_dir}")
        if args.test_dir:
            logging.info(f"Test: {args.test_dir}")
    else:
        logging.info("Step 4: Annotation Builder (Skipped)")

    if not pipeline_config.get('skip_training', False):
        logging.info("Step 5: Training")
        launch_training_in_docker(
            args.train_dir,
            args.val_dir,
            args.checkpoint_output_dir,
            args.training_image,
            args.focus_label,
            args.log_level
        )
    else:
        logging.info("Step 5: Training (Skipped)")

    if not pipeline_config.get('skip_inference', False):
        logging.info("Step 6: Inference")
        launch_inference_in_docker(
            args.inference_weights_path,
            args.inference_test_image,
            args.inference_output_dir,
            args.inference_image,
            args.test_dir,
            args.inference_confidence_threshold,
            args.inference_gradio_mode
        )
    else:
        logging.info("Step 6: Inference (Skipped)")


if __name__ == "__main__":
    main()
