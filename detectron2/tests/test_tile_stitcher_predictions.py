#!/usr/bin/env python3
"""Test script to verify the predictions file matching logic."""
# isort: skip_file
# fmt: off
# ruff: noqa: E402
import os
import tempfile
import shutil
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tiling.tile_stitcher import TileStitcher


def test_predictions_matching():
    """Test that predictions files are correctly matched with inference files."""

    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")

        os.makedirs(input_dir)
        os.makedirs(output_dir)
        os.makedirs(os.path.join(input_dir, "images"))

        # Create a mock tiles_metadata.json
        metadata = [
            {
                "filename": "Hossur_Geratti_2_x4608_y4608.png",
                "site": "Hossur_Geratti_2",
                "pixel_origin": [0, 0],
                "tile_bounds": [0, 0, 100, 100],
                "crs": "EPSG:4326"
            }
        ]

        import json
        with open(os.path.join(input_dir, "tiles_metadata.json"), "w") as f:
            json.dump(metadata, f)

        # Create mock files in output directory
        # Case 1: Both inference and predictions files exist
        with open(os.path.join(output_dir, "inference_Hossur_Geratti_2_x4608_y4608.png"), "w") as f:
            f.write("mock image data")
        with open(os.path.join(output_dir, "predictions_Hossur_Geratti_2_x4608_y4608.json"), "w") as f:
            f.write('{"predictions": []}')

        # Case 2: Inference file exists but predictions file doesn't
        with open(os.path.join(output_dir, "inference_Hossur_Geratti_2_x4608_y4608_missing.png"), "w") as f:
            f.write("mock image data")

        # Case 3: Original tile file (no inference file)
        with open(os.path.join(input_dir, "images", "Hossur_Geratti_2_x4608_y4608.png"), "w") as f:
            f.write("original tile data")

        # Initialize TileStitcher
        stitcher = TileStitcher(input_dir, output_dir)

        # Test the _match_output_file method
        tile_file = "Hossur_Geratti_2_x4608_y4608.png"
        inference_path, predictions_path = stitcher._match_output_file(
            tile_file)

        print(f"Testing tile: {tile_file}")
        print(f"Inference path: {inference_path}")
        print(f"Predictions path: {predictions_path}")

        # Verify results
        expected_inference = os.path.join(
            output_dir, "inference_Hossur_Geratti_2_x4608_y4608.png")
        expected_predictions = os.path.join(
            output_dir, "predictions_Hossur_Geratti_2_x4608_y4608.json")

        assert inference_path == expected_inference, f"Expected {expected_inference}, got {inference_path}"
        assert predictions_path == expected_predictions, f"Expected {expected_predictions}, got {predictions_path}"

        print("Test passed: Both inference and predictions files found correctly")

        # Test with missing predictions file
        tile_file_missing = "Hossur_Geratti_2_x4608_y4608_missing.png"
        inference_path_missing, predictions_path_missing = stitcher._match_output_file(
            tile_file_missing)

        print(f"\nTesting tile with missing predictions: {tile_file_missing}")
        print(f"Inference path: {inference_path_missing}")
        print(f"Predictions path: {predictions_path_missing}")

        expected_inference_missing = os.path.join(
            output_dir, "inference_Hossur_Geratti_2_x4608_y4608_missing.png")

        assert inference_path_missing == expected_inference_missing, f"Expected {expected_inference_missing}, got {inference_path_missing}"
        assert predictions_path_missing is None, f"Expected None for missing predictions, got {predictions_path_missing}"

        print("Test passed: Missing predictions file handled correctly")


if __name__ == "__main__":
    test_predictions_matching()
    print("\nAll tests passed!")
