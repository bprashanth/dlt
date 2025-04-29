"""Discover the relevant files for training from new datasets.

Usage: 
    source venv/bin/activate 
    python3 main.py --root_dir /path/to/root/dir --log_level DEBUG
"""

import os
import logging

# Get module-level logger
logger = logging.getLogger(__name__)


class DataDiscovery:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.tiff_files = self._discover_tiffs()
        self.shapefile = self._discover_shapefile()

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

    def get_discovery_results(self):
        return {
            "tiff_files": self.tiff_files,
            "shapefile": self.shapefile
        }
