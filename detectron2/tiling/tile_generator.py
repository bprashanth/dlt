"""Generate tiles from TIFF files.

Usage: 
    TileGenerator(
        discovery_results, 
        output_dir, 
        tile_size=512, 
        overlap=128).generate_tiles()

Sample discovery results: 
    {
        "tiff_files": ["path/to/tiff1.tif", "path/to/tiff2.tif"],
        "shapefile": "path/to/shapefile.shp",
        "classes": ["class1", "class2"],
        "name_key": "Name"
    }

Sample output: tiles_metadata.json  
    [
        {
            "filename": "Gavihalla_x0_y0.png",
            "site": "Gavihalla",
            "tile_bounds": [x1, y1, x2, y2],
            "pixel_origin": [x, y],
            "crs": "EPSG:32643"
        },
        ...
    ]

Output: 
    output/
    ├── images/
    │   ├── Gavihalla_x0_y0.png
    │   ├── Gavihalla_x384_y0.png
    │   └── ...
    └── tiles_metadata.json
"""

import os
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from shapely.geometry import box
import json
import math
from tqdm import tqdm
import logging

# Get module-level logger
logger = logging.getLogger(__name__)


class TileGenerator:
    """Generate tiles from TIFF files.

    @param discovery_results: The output of the discovery script.
    @param output_dir: The directory to save the tiles.
    @param tile_size: The size of the tiles to generate.
    @param overlap: The overlap between tiles.
    @param max_tiles: The maximum number of tiles to generate.
    @param skip_threshold: The minimum percent of valid pixels required in a tile to be included in the dataset. The default is 10%.
    @param no_tile: If True, preserves each TIFF as a single PNG without tiling.
    """

    def __init__(self, discovery_results, output_dir, tile_size=512, overlap=128, skip_threshold=10.0, no_tile=False):
        self.tiff_files = discovery_results["tiff_files"]
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.overlap = overlap
        self.skip_threshold = skip_threshold
        self.no_tile = no_tile
        self.tile_output_dir = os.path.join(output_dir, "images")
        self.tile_metadata = []
        os.makedirs(self.tile_output_dir, exist_ok=True)

    def generate_tiles(self):
        for tiff_path in tqdm(self.tiff_files, desc="Tiling orthomosaics"):
            self._tile_single_tiff(tiff_path)
        return self._save_metadata()

    def _is_tile_valid(self, mask_array):
        """Check if a tile is valid basis the % of data pixels. 

        @param mask_array (ndarray): Mask cropped to the tile window
            shape = (H, W)

        @returns bool: True if tile should be kept, False otherwise.
        """
        total_pixels = mask_array.size
        valid_pixels = (mask_array > 0).sum()

        valid_percent = (valid_pixels / total_pixels) * 100
        return valid_percent >= self.skip_threshold

    def _tile_single_tiff(self, tiff_path):
        skipped_tiles = []
        site_name = os.path.splitext(os.path.basename(tiff_path))[0]

        with rasterio.open(tiff_path) as src:
            width = src.width
            height = src.height
            transform = src.transform
            crs = src.crs

            if self.no_tile:
                # Handle entire TIFF as one tile
                window = Window(0, 0, width, height)
                tile_transform = src.transform
                tile_bounds = src.bounds
                tile_img = src.read()
                tile_filename = f"{site_name}.png"
                tile_path = os.path.join(self.tile_output_dir, tile_filename)

                with rasterio.open(
                    tile_path,
                    "w",
                    driver="PNG",
                    height=height,
                    width=width,
                    count=src.count,
                    dtype=tile_img.dtype,
                    transform=tile_transform,
                    crs=crs
                ) as dst:
                    dst.write(tile_img)

                self.tile_metadata.append({
                    "filename": tile_filename,
                    "site": site_name,
                    "tile_bounds": list(tile_bounds),
                    "pixel_origin": [0, 0],
                    "crs": crs.to_string(),
                })
                return

            step = self.tile_size - self.overlap

            # Window: (x, y, width, height), a crop box for the tiff
            # Transform: a transformation matrix, map pixel -> geo coordinates.
            #   See Affine transform in rasterio docs.
            # pixel_origin: the pixel coordinates of the tile's origin
            # tile_bounds: the bounds of the tile in geo coordinates
            #   Specifically, these bounds are in the CRS of the tiff.
            #   The units vary by CRS. In UTM, the units are meters.
            #   In lat/lon, the units are degrees.
            #   Tracked in issues/8.
            # The tiff itself has the metadata to go from pixel -> geo.
            for y in range(0, height, step):
                for x in range(0, width, step):
                    window = Window(x, y, self.tile_size, self.tile_size)

                    if x + self.tile_size > width:
                        x = width - self.tile_size
                        window = Window(x, y, self.tile_size, self.tile_size)

                    if y + self.tile_size > height:
                        y = height - self.tile_size
                        window = Window(x, y, self.tile_size, self.tile_size)

                    tile_transform = src.window_transform(window)
                    tile_bounds = rasterio.windows.bounds(window, transform)
                    tile_img = src.read(window=window)
                    tile_filename = f"{site_name}_x{x}_y{y}.png"
                    tile_path = os.path.join(
                        self.tile_output_dir, tile_filename)

                    # This mask contains the nodata value for this tile. It
                    # could be "0", or transparent, depending on the tiff.
                    # image = [
                    #     [10, 10],
                    #     [0, 0]    transparent/nodata pixels
                    # ]
                    # mask = [
                    #     [1, 1],
                    #     [0, 0]    mask shows these as invalid
                    # ]
                    mask = src.read_masks(1)

                    # Crop the mask to the current tile window.
                    tile_mask = mask[
                        window.row_off:window.row_off + window.height,
                        window.col_off:window.col_off + window.width
                    ]

                    if not self._is_tile_valid(tile_mask):
                        skipped_tiles.append(tile_filename)

                    # Overwrite existing tiles.
                    # Since we are writing to PNG, rasterio will also supply a .
                    # aux.xml file with the metadata from the tiff. If this is
                    # not needed, use:
                    # with rasterio.Env(GDAL_PAM_ENABLED="NO"):
                    #   with rasterio.open(..., driver="PNG", ...) as dst:
                    #     dst.write(...)
                    with rasterio.open(
                        tile_path,
                        "w",
                        driver="PNG",
                        height=self.tile_size,
                        width=self.tile_size,
                        count=src.count,
                        dtype=tile_img.dtype,
                        transform=tile_transform,
                        crs=crs
                    ) as dst:
                        dst.write(tile_img)

                    self.tile_metadata.append({
                        "filename": tile_filename,
                        "site": site_name,
                        "tile_bounds": list(tile_bounds),
                        "pixel_origin": [x, y],
                        "crs": crs.to_string(),
                        "is_valid": tile_filename not in skipped_tiles
                    })

        if skipped_tiles:
            logging.info(
                f"Skipped {len(skipped_tiles)} tiles for {site_name}.")
        return

    def _save_metadata(self):
        metadata_path = os.path.join(self.output_dir, "tiles_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.tile_metadata, f, indent=2)
        return metadata_path
