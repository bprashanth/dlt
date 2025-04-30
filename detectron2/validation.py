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
from exceptions import InputException
from exceptions import ValidationException

# Get module-level logger
logger = logging.getLogger(__name__)


class DataValidator:
    """Validate the dataset and collect class information.

    @param discovery_results: The output of the discovery script.

    @return: A list of dictionaries with the following keys:
        - site: The relative path to the site file.
        - area: The area of the site taken from the bounding box of the TIFF. 
        - classes: A dictionary with the class name as the key and the count as the value, of all the classes found in the TIFF (i.e all the intersecting polygons with the shp file and the TIFF perimeter).

    @raises: 
        - InputError: If the discovery results are not valid.
        - ValidationError: If no classes are found in the TIFFs, i.e no TIFF intersects with the shp file. This indicates a mismatch somewhere, either in the crs, or the sites and the shp file. 
    """

    def __init__(self, discovery_results):
        try:
            self.tiff_files = discovery_results["tiff_files"]
            self.shapefile = discovery_results["shapefile"]
            self.name_key = discovery_results["name_key"]
        except KeyError as e:
            raise InputException(f"Invalid discovery results: {e}")

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
                    class_counts = intersecting[self.name_key].value_counts(
                    ).to_dict()
                else:
                    class_counts = {}
                    logging.warning(
                        f"No intersecting polygons found for {tiff_path}")

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

        if not any('classes' in result and result['classes']
                   for result in self.results):
            raise ValidationException("No classes found in validation results")

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
