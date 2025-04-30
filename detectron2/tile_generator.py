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
    """

    def __init__(self, discovery_results, output_dir, tile_size=512, overlap=128):
        self.tiff_files = discovery_results["tiff_files"]
        self.output_dir = output_dir
        self.tile_size = tile_size
        self.overlap = overlap
        self.tile_output_dir = os.path.join(output_dir, "images")
        self.tile_metadata = []
        os.makedirs(self.tile_output_dir, exist_ok=True)

    def generate_tiles(self):
        for tiff_path in tqdm(self.tiff_files, desc="Tiling orthomosaics"):
            self._tile_single_tiff(tiff_path)
        return self._save_metadata()

    def _tile_single_tiff(self, tiff_path):
        site_name = os.path.splitext(os.path.basename(tiff_path))[0]

        with rasterio.open(tiff_path) as src:
            width = src.width
            height = src.height
            transform = src.transform
            crs = src.crs

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
                    })

    def _save_metadata(self):
        metadata_path = os.path.join(self.output_dir, "tiles_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.tile_metadata, f, indent=2)
        return metadata_path
