"""
This script is used to downscale an image.

Usage: 
    python3 downscale_image.py --image ../inference/ \
    inference_Jawal_Lakshmipura_1_x1536_y3072.png \
    --output downscaled.png \
    --scale_factor 0.1
"""

# isort: skip_file
# fmt: off
# ruff: noqa: E402
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tiling.tile_stitcher import TileStitcher


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downscale an image")
    parser.add_argument("--image", required=True, help="Path to the image")
    parser.add_argument("--scale_factor", type=float, default=0.1,
                        help="Scale factor for downscaling")
    parser.add_argument("--output", required=True,
                        help="Path to the output image")
    args = parser.parse_args()

    TileStitcher.generate_preview(args.image, args.output, args.scale_factor)
