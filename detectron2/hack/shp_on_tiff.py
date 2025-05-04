import argparse
import os
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

def plot_tiff_with_colored_shapefile(tiff_path, shp_path, output_path, class_name_key="Name"):
    # Load raster
    with rasterio.open(tiff_path) as src:
        bounds = src.bounds
        crs = src.crs
        count = src.count

        # Read and normalize image (RGB or grayscale)
        if count >= 3:
            image = src.read([1, 2, 3]).transpose(1, 2, 0)
        else:
            band = src.read(1)
            image = np.stack([band]*3, axis=-1)

        image = image.astype(float)
        image_min, image_max = image.min(), image.max()
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)

    # Load shapefile
    gdf = gpd.read_file(shp_path)
    if class_name_key not in gdf.columns:
        raise ValueError(f"Column '{class_name_key}' not found in shapefile.")

    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Clip to raster bounds
    raster_box = box(*bounds)
    gdf = gdf[gdf.intersects(raster_box)]

    # Generate unique colors per class
    class_names = sorted(gdf[class_name_key].unique())
    color_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    color_map = {cls: color_list[i % len(color_list)] for i, cls in enumerate(class_names)}

    # Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])

    # Plot each class in its color
    for cls_name in class_names:
        subset = gdf[gdf[class_name_key] == cls_name]
        subset.boundary.plot(ax=ax, edgecolor=color_map[cls_name], linewidth=1, label=cls_name)

    # Create legend
    handles = [mpatches.Patch(color=color_map[cls], label=cls) for cls in class_names]
    ax.legend(handles=handles, title="Class", loc='lower right', fontsize='small', title_fontsize='medium')
    ax.set_title("TIFF with Class-Colored Shapefile Overlay")
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved visualization to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay shapefile polygons on TIFF with class-based coloring.")
    parser.add_argument("--tiff", required=True, help="Path to the TIFF image")
    parser.add_argument("--shp", required=True, help="Path to the .shp file (other files must be in the same directory)")
    parser.add_argument("--output", default="overlay_output.png", help="Path to save the output PNG")
    parser.add_argument("--class_name_key", default="Name", help="Column in the shapefile representing class labels")

    args = parser.parse_args()
    plot_tiff_with_colored_shapefile(args.tiff, args.shp, args.output, args.class_name_key)

