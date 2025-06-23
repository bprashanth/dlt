import geopandas as gpd
from shapely.geometry import box


class BoundsConverter:
    """Converts bounds from one CRS to another."""
    GPS_CRS = "EPSG:4326"
    WEB_MERCATOR_CRS = "EPSG:3857"

    def __init__(self, bounds, crs, target_crs=GPS_CRS):
        """
        @param bounds: List of bounds in the source CRS.
        @param crs: Source CRS.
        @param target_crs: Target CRS. GPS=EPSG:4326, Web Mercator=EPSG:3857.
        """
        if not crs:
            raise ValueError("CRS must be provided for bounds conversion.")
        self.bounds = bounds
        self.crs = crs
        self._geom = box(*bounds)
        self._gdf = gpd.GeoSeries([self._geom], crs=crs)
        self.target_crs = target_crs

    def get_bounds(self):
        """Return bounds in the target CRS as [minx, miny, maxx, maxy]"""
        try:
            gdf_proj = self._gdf.to_crs(self.target_crs)
            return list(gdf_proj.total_bounds)
        except Exception as e:
            raise RuntimeError(f"CRS transformation failed: {e}")

    def get_center(self):
        """Return centroid in the target CRS as [lon, lat] or [x, y]"""
        try:
            gdf_proj = self._gdf.to_crs(self.target_crs)
            centroid = gdf_proj.geometry[0].centroid
            return [centroid.x, centroid.y]
        except Exception as e:
            raise RuntimeError(f"CRS transformation failed: {e}")
