"""Discover the relevant files for training from new datasets.

Usage: 
    DataDiscovery(root_dir).get_discovery_results()

Returns: 
{
    "tiff_files": [list of paths to tiff files in root dir],
    "shapefile": path to shapefile in root dir,
    "class_map": {class name: class id},
    "name_key": [name of the key in the shapefile that contains the class names]
}
"""

import os
import logging
import geopandas as gpd
import numpy as np
import pandas as pd

# Get module-level logger
logger = logging.getLogger(__name__)


class DataDiscovery:
    """Discover the relevant files for training from new datasets.

    @param root_dir: The root directory to search for data.
    @param name_key: The key in the shapefile that contains the class names.

    @return: A dictionary with the following keys:
        - tiff_files: A list of paths to the site .tiff files.
        - shapefile: The path to the shapefile.
        - classes: A list of class names.   
    """

    def __init__(self, root_dir, name_key="Name", id_key="Type"):
        self.root_dir = root_dir
        self.tiff_files = self._discover_tiffs()
        self.shapefile = self._discover_shapefile()
        self.name_key = name_key
        self.id_key = id_key
        self.classes = self._discover_classes()

    def _discover_tiffs(self):
        """Recursively find all site .tiff files under root_dir.

        This function matches site directories with site.tiff files and ignores all other tiff files. Eg: 

        /foo/bar/site_1/site_1.tiff
        /foo/bar/site_1/Bamboo/bamboo_123.tiff
        /foo/bar/site_1/Bamboo/bamboo.tiff

        This function will return: 
        /foo/bar/site_1/site_1.tiff
        /foo/bar/site_1/Bamboo/bamboo.tiff
        """
        main_tiffs = []

        for subdir, dirs, files in os.walk(self.root_dir):
            dir_name_raw = os.path.basename(subdir)
            dir_name_normalized = dir_name_raw.lower().replace(" ", "_")

            for file in files:
                if file.lower().endswith(".tiff"):
                    file_stem = os.path.splitext(file)[0].lower()
                    if file_stem == dir_name_normalized:
                        main_tiffs.append(os.path.join(subdir, file))
        return main_tiffs

    def _discover_shapefile(self):
        """Discover the first shapefile found in the root_dir."""
        shp_files = [f for f in os.listdir(
            self.root_dir) if f.endswith(".shp")]
        if not shp_files:
            raise FileNotFoundError(f"No shapefiles found in {self.root_dir}")
        return os.path.join(self.root_dir, shp_files[0])

    def _discover_classes(self):
        """Discover the classes and their IDs in the shapefile. 

        @return: A dictionary with the class name as the key and the class ID as the value.

        @raises: 
            ValueError: If required columns are missing of it IDs are invalid. 
        """
        # TODO(prashanth@): this should NOT be taken from the shapefile. We
        # should have an internal mapping of class names to IDs.
        gdf = gpd.read_file(self.shapefile)

        if self.name_key not in gdf.columns:
            raise ValueError(
                f"Shapefile {self.shapefile} does not contain a {self.name_key} column")

        if self.id_key not in gdf.columns:
            raise ValueError(
                f"Shapefile {self.shapefile} does not contain a {self.id_key} column")

        # Create initial name->id mapping from unique combinations
        class_map = dict(zip(gdf[self.name_key], gdf[self.id_key]))

        # Remove NaN/None entries
        class_map = {k: v for k, v in class_map.items() if pd.notna(k)
                     and pd.notna(v)}

        # Verify all IDs are integers
        for name, id_val in class_map.items():
            if not isinstance(id_val, (int, np.integer)):
                raise ValueError(
                    f"Invalid ID type for class {name}: {type(id_val)}. Must be integer.")

        # Adjust IDs to start at 1 for COCO compatibility
        min_id = min(class_map.values())
        adjustment = 1 - min_id if min_id < 1 else 0

        if adjustment != 0:
            logger.info(
                f"Adjusting class IDs by +{adjustment} to ensure COCO compatibility (ids start at 1)")
            class_map = {name: id_val + adjustment for name,
                         id_val in class_map.items()}

        return class_map

    def get_discovery_results(self):
        return {
            "tiff_files": self.tiff_files,
            "shapefile": self.shapefile,
            "class_map": self.class_map,
            "name_key": self.name_key,
            "id_key": self.id_key
        }
