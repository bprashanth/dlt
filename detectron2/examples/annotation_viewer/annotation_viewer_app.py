import streamlit as st
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import json
from shapely.geometry import shape
import cv2
import os


import streamlit as st
import rasterio
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt


def plot_shapefile_on_tiff(tiff_path, shp_path):
    st.write("### Shapefile over TIFF")

    # Load raster
    with rasterio.open(tiff_path) as src:
        bounds = src.bounds
        crs = src.crs
        count = src.count

        # Handle RGB or single-band fallback
        if count >= 3:
            image = src.read([1, 2, 3])  # RGB
            image = image.transpose(1, 2, 0)
        else:
            band = src.read(1)
            image = np.stack([band] * 3, axis=-1)

        # Normalize image to [0, 1]
        image = image.astype(float)
        image_min, image_max = image.min(), image.max()
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)

    # Load and align shapefile
    gdf = gpd.read_file(shp_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Plot the image and overlay polygons
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, extent=[bounds.left,
              bounds.right, bounds.bottom, bounds.top])
    gdf.boundary.plot(ax=ax, edgecolor='red', linewidth=1)
    ax.set_title("TIFF with Shapefile Overlay")
    ax.axis('off')

    st.pyplot(fig)


def plot_coco_on_tile(png_path, coco_path):
    st.write("### COCO Annotations over Tile PNG")

    # Load image
    image = cv2.imread(png_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # Load COCO JSON
    with open(coco_path, "r") as f:
        coco = json.load(f)

    # Find image entry
    image_filename = os.path.basename(png_path)
    image_id = None
    for img in coco["images"]:
        if os.path.basename(img["file_name"]) == image_filename:
            image_id = img["id"]
            break

    if image_id is None:
        st.warning("Image not found in COCO JSON.")
        return

    # Get annotations
    anns = [a for a in coco["annotations"] if a["image_id"] == image_id]

    # Draw annotations
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    for ann in anns:
        for seg in ann["segmentation"]:
            xs = seg[::2]
            ys = seg[1::2]
            ax.plot(xs, ys, color='lime', linewidth=2)
    ax.set_title(f"Annotations for {image_filename}")
    st.pyplot(fig)


def save_uploaded(file, suffix):
    if file:
        with open(f"temp.{suffix}", "wb") as f:
            f.write(file.read())
        return f"temp.{suffix}"
    return None


# Streamlit layout
st.set_page_config(layout="wide")
st.title("Annotation Comparison Viewer")

col1, col2 = st.columns(2)

with col1:
    st.header("Raw Data View")
    tiff_file = st.file_uploader("Upload a TIFF File", type=["tiff", "tif"])

    st.header("Upload Shapefile Components")

    uploaded_shp = st.file_uploader("Upload .shp file", type="shp")
    uploaded_shx = st.file_uploader("Upload .shx file", type="shx")
    uploaded_dbf = st.file_uploader("Upload .dbf file", type="dbf")
    uploaded_prj = st.file_uploader("Upload .prj file (optional)", type="prj")
    uploaded_cpg = st.file_uploader("Upload .cpg file (optional)", type="cpg")

    shp_path = save_uploaded(uploaded_shp, "shp")
    save_uploaded(uploaded_shx, "shx")
    save_uploaded(uploaded_dbf, "dbf")
    save_uploaded(uploaded_prj, "prj")
    save_uploaded(uploaded_cpg, "cpg")

    if shp_path and uploaded_shx and uploaded_dbf:
        plot_shapefile_on_tiff("temp.tiff", shp_path)


with col2:
    st.header("COCO View")
    png_file = st.file_uploader("Upload a PNG Tile", type=["png"])
    coco_json = st.file_uploader("Upload COCO JSON", type=["json"])

    if png_file and coco_json:
        with open("temp.png", "wb") as f:
            f.write(png_file.read())
        with open("temp.json", "wb") as f:
            f.write(coco_json.read())
        plot_coco_on_tile("temp.png", "temp.json")
