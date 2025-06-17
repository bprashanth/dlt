"""Stitch tiles into a single image. 

Usage: 
    TileStitcher(
        input_tile_dir, 
        output_tile_dir, 
        output_format='png').stitch()

Sample input: 
    input_tile_dir/
    ├── images/
    │   ├── Gavihalla_x0_y0.png
    │   ├── Gavihalla_x384_y0.png
    └── tiles_metadata.json

    output_tile_dir/
    ├── inference_Gavihalla_x0_y0.png
    ├── ...    
    └── inference_Gavihalla_x384_y0.png

Sample output: 
    output_tile_dir/
    ├── inference_Gavihalla_x0_y0.png
    ├── ...
    ├── inference_Gavihalla_x384_y0.png    
    └── Gavihalla_stitched.png

For example to simply re-stitch all the input tiles, you can run: 
TileStitcher(
    input_tile_dir="input_tile_dir",
    output_tile_dir="input_tile_dir/images",
    output_format="png"
).stitch()

And look for the stitched site images in the output_tile_dir/images directory. 

In other words the output_tile_dir is both the base directory for inputs *into* the final mosaic, as well as the directory into which the final mosaic is saved. If the tiles are not found in the output_tile_dir, the stitcher will look for them in the input_tile_dir/images directory. 
"""

import os
import json
from PIL import Image
from tqdm import tqdm
import logging
from collections import defaultdict

# Get module-level logger
logger = logging.getLogger(__name__)


class TileStitcher:
    def __init__(self, input_tile_dir, output_tile_dir, output_format='png'):
        self.input_tile_dir = input_tile_dir
        self.output_tile_dir = output_tile_dir
        self.output_format = output_format

        self.metadata_path = os.path.join(
            self.input_tile_dir, "tiles_metadata.json")
        self.input_images_dir = os.path.join(self.input_tile_dir, "images")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Missing tiles_metadata.json at {self.metadata_path}")

        self.tile_metadata = self._load_metadata()
        self.tile_size = None
        self.step_size = None
        self.output_image_files = [
            f for f in os.listdir(self.output_tile_dir)]

    def _load_metadata(self):
        with open(self.metadata_path, 'r') as f:
            return json.load(f)

    def _get_tile_size(self):
        """Get the size of tiles in the input directory. 

        This function makes a few assumptions: 
          1. All tiles are the same size (all input tiles, and input tiles are the same size as output tiles). 
          2. All tiles are square. 

        Returns: 
            tile_size: int, the size of the tiles in the input directory. 
        """
        for tile in self.tile_metadata:
            tile_path = os.path.join(self.input_images_dir, tile['filename'])
            if os.path.exists(tile_path):
                with Image.open(tile_path) as img:
                    self.tile_size = img.size[0]
                    return
        return RuntimeError("No valid tiles found in input directory to determine tile size.")

    def _get_canvas_size(self, tiles):
        max_x, max_y = 0, 0
        for tile in tiles:
            x, y = tile['pixel_origin']
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        width = max_x + self.tile_size
        height = max_y + self.tile_size
        return width, height

    def _match_output_file(self, original_filename):
        # Match based on suffix: find file ending in original_filename.
        for fname in self.output_image_files:
            if fname.endswith(original_filename):
                return os.path.join(self.output_tile_dir, fname)
        return None

    def stitch(self):
        logger.info("Loading metadata and determining tile layout...")
        self._get_tile_size()
        logger.debug(f"Tile size: {self.tile_size}")

        # Group tiles by site
        site_tiles = defaultdict(list)
        for tile in self.tile_metadata:
            site_tiles[tile['site']].append(tile)

        flat_metadata = []

        for site_name, tiles in site_tiles.items():
            logger.info(f"\nStitching site: {site_name}")
            stitched_width, stitched_height = self._get_canvas_size(tiles)
            canvas = Image.new('RGB', (stitched_width, stitched_height))

            tile_index = 0
            for tile in tqdm(tiles):
                x, y = tile['pixel_origin']
                tile_file = tile['filename']

                # Prefer the file in the output_dir and fallback to the
                # original tile if not found.
                output_tile_path = self._match_output_file(tile_file)
                tile_path = output_tile_path

                if not tile_path or not os.path.exists(tile_path):
                    tile_path = os.path.join(self.input_images_dir, tile_file)

                if not os.path.exists(tile_path):
                    logger.warning(
                        f"Warning: Missing tile at {tile_path}, skipping...")
                    continue

                with Image.open(tile_path) as tile_img:
                    canvas.paste(tile_img, (x, y))

                flat_metadata.append({
                    "image": os.path.relpath(tile_path, self.output_tile_dir),
                    "tile_origin": [x, y],
                    "tile_index": tile_index,
                    "site": {
                        "name": site_name,
                        "preview": f"{site_name}_stitched.png"
                    }
                })
                tile_index += 1

            output_path = os.path.join(
                self.output_tile_dir, f"{site_name}_stitched.png")
            canvas.save(output_path)
            logger.info(f"Stitched image saved to: {output_path}")

        inference_metadata_path = os.path.join(
            self.output_tile_dir, "inference_tile_metadata.json")
        with open(inference_metadata_path, 'w') as f:
            json.dump(flat_metadata, f, indent=2)
        logger.info(f"Inference metadata saved to: {inference_metadata_path}")

        return inference_metadata_path
