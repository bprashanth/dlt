"""Validate new datasets and extract classes by intersecting polygons. 

This scripts works on the output of the discovery script. 

1. For every tiff discovered, read it's bounding box. 
2. Look through every polygon in the discovered shp file for intersecting polygons 
3. For each intersecting polygon, register the class name as belonging to that site 

Usage: 
    source venv/bin/activate 
    python3 main.py --root_dir /path/to/root/dir --log_level DEBUG
"""

import os
import geopandas as gpd
import rasterio
from shapely.geometry import box
import fiona
import logging

# Get module-level logger
logger = logging.getLogger(__name__)


class DataValidator:
    def __init__(self, discovery_results):
        self.tiff_files = discovery_results["tiff_files"]
        self.shapefile = discovery_results["shapefile"]
        self.gdf = gpd.read_file(self.shapefile)
        self.results = []
        self.all_classes = {}

    def _process_tiff(self, tiff_path):
        """For a given TIFF, find intersecting  polygons and return site info."""
        rel_path = os.path.relpath(tiff_path, os.path.dirname(self.shapefile))

        try:
            with rasterio.open(tiff_path) as src:
                bounds = src.bounds
                tiff_polygon = box(*bounds)

                if self.gdf.crs != src.crs:
                    gdf_projected = self.gdf.to_crs(src.crs)
                else:
                    gdf_projected = self.gdf

                # Find polygons intersecting this TIFF's bounds
                intersecting = gdf_projected[gdf_projected.intersects(
                    tiff_polygon)]

                if not intersecting.empty:
                    class_counts = intersecting['Name'].value_counts(
                    ).to_dict()
                else:
                    class_counts = {}

                area = tiff_polygon.area

                return {
                    "site": rel_path,
                    "area": area,
                    "classes": class_counts
                }
        except Exception as e:
            logger.error(f"Error processing {tiff_path}")
            logger.error(
                f"Intersecting object: {intersecting if 'intersecting' in locals() else 'not created'}")
            return {
                "site": rel_path,
                "error": str(e)
            }

    def validate(self):
        """Validate the dataset and collect class information."""
        logger.info(f"Processing {len(self.tiff_files)} TIFF files")
        self.results = []
        self.all_classes = {}

        for tiff_path in self.tiff_files:
            site_info = self._process_tiff(tiff_path)
            self.results.append(site_info)

            if 'classes' in site_info:
                for class_name, count in site_info['classes'].items():
                    self.all_classes[class_name] = self.all_classes.get(
                        class_name, 0) + count

        return self.results

    def get_classes(self):
        """Return a list of all classes found in the dataset."""
        if not self.all_classes:
            logger.warning("No classes found. Run validate() first.")
            return []

        all_classes_counted = [(class_name, count)
                               for class_name, count in self.all_classes.items()]
        all_classes_counted.sort(key=lambda x: x[1], reverse=True)
        return all_classes_counted


def inspect_formats(shp_path, tiff_files):
    """Load and inspect the shapefile and tiff files, printing the schema and a few sample records."""
    gdf = gpd.read_file(shp_path)
    schema = fiona.open(shp_path).schema
    logger.debug(f"Shapefile schema: {schema}")
    logger.debug(f"Sample records: {gdf.head()}")
    logger.debug(f"Shapefile CRS: {gdf.crs}")
    with rasterio.open(tiff_files[0]) as src:
        logger.debug(f"TIFF file: {tiff_files[0]}")
        logger.debug(f"TIFF metadata: {src.meta}")
        logger.debug(f"TIFF CRS: {src.crs}")
        if src.crs != gdf.crs:
            logger.warning(
                f"Shapefile and TIFF CRS do not match: {gdf.crs} != {src.crs}")
