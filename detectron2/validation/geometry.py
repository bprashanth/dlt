"""Validates and fixes geometries in a GeoDataFrame.

Usage: 

1. Validate and fix geometries

    validator = GeometryValidator(gdf)
    validator.fix()

2. Validate only

    validator = GeometryValidator(gdf)
    validator.validate(strict=True)
"""

import logging
from shapely.validation import explain_validity
import geopandas as gpd
from shapely.validation import make_valid

# Get module-level logger
logger = logging.getLogger(__name__)


class GeometryValidator:
    """Handles validation and fixing of geometries in GeoDataFrames."""

    def __init__(self, gdf):
        """Initialize with a GeoDataFrame.

        Args:
            gdf (GeoDataFrame): The GeoDataFrame to validate/fix
        """
        self.gdf = gdf.copy()
        self.logger = logging.getLogger(__name__)

    def _check_null(self, strict=False):
        """Check for null geometries in gdf.

        A null geometrys is a row in the df that has a label but no polygon. Eg: 

        >>> gdf = gpd.read_file(shp)
        >>> gdf[gdf.geometry.isna()]

        id  Types           Name             Place geometry
        702  NaN      4  Lantana Cover  Hossur Geratti 1     None
        3343 NaN      5          Trees     Tholuvu Betta     None

        The indicates that indices 702 and 3343 have null geometries. 
        The annotation builder discards these, but we flag them here so the user can fix them. 

        @returns: GeoDataFrame containing only the null geometries.
        """
        null_geoms = self.gdf[self.gdf.geometry.isna()]
        if not null_geoms.empty:
            if strict:
                raise ValueError(
                    f"Found {len(null_geoms)} null geometries")
            else:
                self.logger.warning(
                    f"Found {len(null_geoms)} null geometries:")
                for idx, row in null_geoms.iterrows():
                    self.logger.warning(f"Row {idx} has null geometry")
        return null_geoms

    def _check_invalid(self, strict=False):
        """Check for invalid geometries in the GeoDataFrame.


        A valid geometry is one that can be plotted without errors. See docs/geometry.md for more details.

        @returns: GeoDataFrame containing only the invalid geometries
        """
        invalid_geoms = self.gdf[~self.gdf.geometry.is_valid]
        if not invalid_geoms.empty:
            if strict:
                raise ValueError(
                    f"Found {len(invalid_geoms)} invalid geometries")
            else:
                self.logger.info(
                    f"Found {len(invalid_geoms)} invalid geometries, run with --log_level DEBUG to see details")
                for idx, row in invalid_geoms.iterrows():
                    if row.geometry is not None:
                        reason = explain_validity(row.geometry)
                        self.logger.debug(f"Row {idx}: {reason}")
                        self.logger.debug(
                            f"Geometry type: {row.geometry.geom_type}")
                        self.logger.debug(
                            f"Geometry WKT: {row.geometry.wkt[:100]}...")
                    else:
                        self.logger.debug(f"Row {idx}: Null geometry")
        return invalid_geoms

    def _fix_null(self):
        """Remove null geometries from the GeoDataFrame."""
        self.gdf = self.gdf.dropna(subset=['geometry'])

    def _fix_invalid(self):
        """Attempt to fix invalid geometries using multiple methods."""
        self.logger.info("Attempting to fix invalid geometries...")

        # Method 1: buffer(0)
        self.gdf['geometry'] = self.gdf.geometry.buffer(0)

        # Method 2: make_valid() for any remaining invalid geometries
        still_invalid = self.gdf[~self.gdf.geometry.is_valid]
        if not still_invalid.empty:
            self.logger.warning(
                f"Buffer(0) left {len(still_invalid)} invalid geometries, trying make_valid()...")
            self.gdf.loc[~self.gdf.geometry.is_valid, 'geometry'] = \
                self.gdf[~self.gdf.geometry.is_valid].geometry.make_valid()

    def validate(self, strict=False):
        """Validate the GeoDataFrame geometries.

        @param strict (bool): If True, raises exceptions for invalid geometries.
            If False, only logs errors.

        @returns: bool: True if valid, False if invalid

        @raises:
            ValueError: If strict=True and invalid geometries are found
        """
        null_geoms = self._check_null()
        invalid_geoms = self._check_invalid()

        has_errors = not (null_geoms.empty and invalid_geoms.empty)

        if strict and has_errors:
            raise ValueError(
                f"Found {len(null_geoms)} null and {len(invalid_geoms)} invalid geometries")

        return not has_errors

    def fix(self, strict_fix=False):
        """Attempt to fix all geometry issues in the GeoDataFrame.

        @param strict_fix (bool): If True, raise an error if geometries cannot be fixed. If False, only log errors.

        @returns: GeoDataFrame: A fixed copy of the input GeoDataFrame

        @raises:
            ValueError: If geometries cannot be fixed
        """
        if self.validate(strict=False):
            return self.gdf

        self._fix_null()
        self._fix_invalid()

        # Final validation
        final_invalid = self.gdf[~self.gdf.geometry.is_valid]
        if not final_invalid.empty:
            if strict_fix:
                raise ValueError(
                    f"Still have {len(final_invalid)} invalid geometries after all fixes")
            else:
                self.logger.error(
                    f"Still have {len(final_invalid)} invalid geometries after all fixes")
                self.gdf = self.gdf[self.gdf.geometry.is_valid]
            self.logger.warning("Proceeding with only valid geometries")
        else:
            self.logger.info("All geometries are now valid")

        return self.gdf

    @staticmethod
    def fix_invalid_geometry(geom):
        """Attempts to fix an invalid geometry using multiple approaches.

        This is used to fix ad hoc geometries that were created by intersection and ended up being invalid. 

        @param geom: A shapely geometry object

        @returns: Fixed geometry if successful, None if all repair attempts fail
        """
        if geom.is_valid:
            return geom

        try:
            # First try make_valid() as it's generally more precise
            fixed = make_valid(geom)
            if fixed.is_valid:
                return fixed

            # If make_valid fails, try zero buffer
            fixed = geom.buffer(0)
            if fixed.is_valid:
                return fixed

            # Last resort - try a tiny positive buffer
            fixed = geom.buffer(0.0001)
            if fixed.is_valid:
                return fixed

            return None

        except Exception as e:
            logger.error(f"Failed to fix invalid geometry: {e}")
            return None
