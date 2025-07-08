import os
import json
import logging
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import transform
from collections import defaultdict
from skimage.draw import polygon as sk_polygon
from pyproj import Transformer

logger = logging.getLogger(__name__)


class MapMetrics:
    """Computes metrics for a map from stitched tile metadata.

    @param input_metadata_path: Path to the stitched tile metadata file
    @param output_metrics_path: Path to the output metrics file
    @param tile_size: Size of the tiles in pixels
    """

    def __init__(self, input_metadata_path, output_metrics_path, tile_size=2048):
        self.input_metadata_path = input_metadata_path
        self.output_metrics_path = output_metrics_path
        self.tile_size = tile_size
        self.tile_metadata = self.load_metadata()
        self.sites = {}

    def load_metadata(self):
        """Loads the tile metadata from the input path.

        This is typically a post stitching metadata file. Either offloaded 
        metadata or stitched metadata. 

        @return: List of tile metadata
        """
        logger.info(f"Loading tile metadata from {self.input_metadata_path}")
        with open(self.input_metadata_path, 'r') as f:
            return json.load(f)

    def compute_canvas_size(self, site_tiles):
        """Compute dimensions of the canvas.

        Takes the x-most tile's pixel_origin and adds the tile size to get the canvas size. Same for y-most tile.

        @param site_tiles: List of tiles for a site
        @return: Tuple of (width, height)
        """
        max_x = max(
            tile['image']['pixel_origin'][0] for tile in site_tiles) + self.tile_size
        max_y = max(
            tile['image']['pixel_origin'][1] for tile in site_tiles) + self.tile_size
        logger.debug(f"Computed canvas size: width={max_x}, height={max_y}")
        # numpy shape is (rows, cols), i.e (y, x)
        return max_x, max_y

    def process(self):
        """Process the map and compute metrics.

        This is the main entry point for the class. 

        @return: Path of the output metrics file.
        """
        site_tiles_map = defaultdict(list)
        for tile in self.tile_metadata:
            parent_name = tile["parent"]["name"]
            site_tiles_map[parent_name].append(tile)

        for site_name, site_tiles in site_tiles_map.items():
            logger.info(f"Processing site: {site_name}")
            self.process_site(site_name, site_tiles)

        return self.write_output()

    def process_site(self, site_name, site_tiles):
        """Process a site and compute metrics

        ASSUMES: 
        - All tiles under a parent have same size
        - All tiles have embedded metadta about the parent site 
        - This embedded parent metadata is the same across tiles in a site

        Assuming these, it consideres the 0th parent metadata for computing 
        area. 

        @param site_name: Name of the site
        @param site_tiles: List of tiles for the site
        """
        height, width = self.compute_canvas_size(site_tiles)
        class_masks = defaultdict(
            lambda: np.zeros((height, width), dtype=bool))

        for tile in site_tiles:
            prediction_path = tile["image"].get("predictions")
            if not prediction_path or not os.path.exists(prediction_path):
                logger.warning(
                    f"Skipping tile with missing predictions: {prediction_path}")
                continue

            with open(prediction_path, 'r') as f:
                predictions = json.load(f)

            origin_x, origin_y = tile["image"]["pixel_origin"]
            for ann in predictions:
                category = ann["category_name"]
                for seg in ann["segmentation"]:
                    coords = list(zip(seg[::2], seg[1::2]))
                    shifted_coords = [
                        (x + origin_x, y + origin_y) for x, y in coords]
                    rr, cc = self.rasterize_polygon(
                        shifted_coords, height, width)
                    # Set the pixels at the row and column indices (i.e pixels
                    # within the shifted_coords polygon) to True
                    class_masks[category][rr, cc] = True

        site_area_m2 = self.compute_size_area(
            site_tiles[0]["parent"]["image"]["bounds"],
            site_tiles[0]["parent"]["image"]["crs"])
        stats = {}
        for category, mask in class_masks.items():
            pixel_count = np.count_nonzero(mask)
            pixel_fraction = pixel_count / (height * width)
            area_m2 = pixel_fraction * site_area_m2
            stats[f"{category}_area"] = area_m2
            stats[f"{category}_percent"] = (area_m2 / site_area_m2) * 100

            logger.info(
                f"{category}: {area_m2:.2f}m², {stats[f'{category}_percent']:.2f}%")

        self.sites[site_name] = {
            "name": site_name,
            "image": site_tiles[0]["parent"]["image"],
            "stats": stats
        }

    def rasterize_polygon(self, coords, height, width):
        """Rasterize a polygon into a binary mask

        @param coords: List of (x, y) coordinates
        @param height: Height of the canvas
        @param width: Width of the canvas
        @return: Tuple of (rr, cc) indices
        """
        if len(coords) < 3:
            return np.array([], dtype=int), np.array([], dtype=int)

        # Separate x and y coordinates to 2 lists
        x, y = zip(*coords)
        # Compute all pixel coords that fall within the polygon
        # rr and cc are the row and column indices of the pixels
        rr, cc = sk_polygon(y, x)
        # Clip the indices to the canvas size to prevent out of bounds errors
        rr = np.clip(rr, 0, height - 1)
        cc = np.clip(cc, 0, width - 1)
        return rr, cc

    def compute_size_area(self, bounds, source_crs="EPSG:4326"):
        """Compute the area of a site in square meters

        @param bounds: List of [min_x, min_y, max_x, max_y]
        @return: Area in square meters
        """
        lon_min, lat_min, lon_max, lat_max = bounds
        polygon = Polygon([
            (lon_min, lat_min),
            (lon_max, lat_min),
            (lon_max, lat_max),
            (lon_min, lat_max),
            (lon_min, lat_min)
        ])
        return self.project_and_area(polygon, source_crs)

    def project_and_area(self, polygon, source_crs="EPSG:4326"):
        """Project the polygon to the canvas and compute the area

        The main operation here is a CRS transformation. If the input is a 
        "degrees" CRS like GPS, we can't use it to accurately compute distances 
        because the distance between degrees varies with latitude. 

        @param polygon: Shapely polygon
        @param source_crs: Source CRS of the polygon
        @return: Area in square meters
        """
        transformer = Transformer.from_crs(
            source_crs, "EPSG:3857", always_xy=True)
        projected = transform(transformer.transform, polygon)
        return projected.area

    def write_output(self):
        """Write the metrics to a JSON file

        @return: Path to the output file
        """
        logger.info(f"Writing site metrics to {self.output_metrics_path}")
        with open(self.output_metrics_path, 'w') as f:
            json.dump(list(self.sites.values()), f, indent=2)
        return self.output_metrics_path
