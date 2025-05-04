import argparse
import json
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import box, Polygon, MultiPolygon


def load_tile_metadata(path):
    with open(path) as f:
        return json.load(f)


def plot_over_tiff(tiff_path, metadata_path, shp_path, output_path, show_label, class_name_key):
    tile_metadata = load_tile_metadata(metadata_path)

    with rasterio.open(tiff_path) as src:
        bounds = src.bounds
        crs = src.crs
        image = src.read([1, 2, 3]).transpose(1, 2, 0)  # RGB
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    # Load and align shapefile
    gdf = gpd.read_file(shp_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Filter for specified label
    if show_label:
        gdf = gdf[gdf[class_name_key] == show_label]

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, extent=extent)
    ax.set_title("TIFF with Tile Boxes and Shapefile Annotations")

    # --- Plot tile boxes ---
    for tile in tile_metadata:
        tile_bounds = tile["tile_bounds"]
        tile_box = box(*tile_bounds)
        if not tile_box.intersects(box(*extent)):
            continue

        rect = patches.Rectangle(
            (tile_bounds[0], tile_bounds[1]),
            tile_bounds[2] - tile_bounds[0],
            tile_bounds[3] - tile_bounds[1],
            linewidth=1,
            edgecolor='cyan',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(tile_bounds[0], tile_bounds[3],
                tile["filename"], fontsize=6, color='cyan')

    # --- Draw shapefile polygons using matplotlib.patches ---
    for geom in gdf.geometry:
        if geom is None:
            continue

        if isinstance(geom, Polygon):
            polys = [geom]
        elif isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)
        else:
            continue

        for poly in polys:
            coords = list(poly.exterior.coords)
            patch = patches.Polygon(
                coords, linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(patch)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff", required=True, help="Path to TIFF")
    parser.add_argument("--tile_metadata", required=True,
                        help="Path to tile metadata JSON")
    parser.add_argument("--shp", required=True, help="Path to .shp file")
    parser.add_argument("--output", default="overlay.png",
                        help="Output PNG path")
    parser.add_argument("--show_label", default="Lantana Cover",
                        help="Only show polygons with this label (e.g., 'Lantana Cover')")
    parser.add_argument("--class_name_key", type=str, default="Name",
                        help="The key in the shapefile that contains the class names")
    args = parser.parse_args()

    plot_over_tiff(args.tiff, args.tile_metadata,
                   args.shp, args.output, args.show_label, args.class_name_key)
