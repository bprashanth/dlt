"""Discover the relevant files for training from new datasets.

Usage: 
    DataDiscovery(root_dir).get_discovery_results()
"""

import os
import logging
import geopandas as gpd

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

    def __init__(self, root_dir, name_key="Name"):
        self.root_dir = root_dir
        self.tiff_files = self._discover_tiffs()
        self.shapefile = self._discover_shapefile()
        self.name_key = name_key
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
        """Discover the classes in the shapefile."""
        gdf = gpd.read_file(self.shapefile)
        if self.name_key not in gdf.columns:
            raise ValueError(
                f"Shapefile {self.shapefile} does not contain a {self.name_key} column")
        return sorted(gdf[self.name_key].dropna().unique().tolist())

    def get_discovery_results(self):
        return {
            "tiff_files": self.tiff_files,
            "shapefile": self.shapefile,
            "classes": self.classes,
            "name_key": self.name_key
        }
