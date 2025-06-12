"""Build COCO-style annotations from discovery results and tile metadata.

Usage:
    AnnotationBuilder(discovery_results, tile_output_dir, tile_metadata_path, train_dir, val_dir, val_split=0.2, seed=42).run()

A few things to note:
1. Directory and path structure 

Training code expects annotation.json file in the following structure:
    <train_data>/annotations.json
    <train_data>/images/
    <val_data>/annotations.json
    <val_data>/images/

And each annotation.json file is a COCO-style JSON file with the following structure:
    "images": [
        {
            "id": 1,
            "file_name": "images/Gavihalla_x0_y0.png",
            "width": 512,
            "height": 512
        }
    ]

2. Annotation Indices (category ids)

* COCO category_id values start at 1 (COCO default)
* AnnotationBuilder creates category IDs starting from 1
* The training code expects category IDs to start at 1, because it converts them to 0-based indices for Detectron2

Unfortunately Detectron2 only supports 0-based indices for category IDs so this part must be handled with care. 
"""

import os
import json
import shutil
from shapely.geometry import box
import geopandas as gpd
from shapely.affinity import affine_transform
from rasterio.transform import Affine
from shapely import affinity as shapely_affinity

from tqdm import tqdm
import random
import logging
from PIL import Image
from validation import GeometryValidator
import PIL

# Get module-level logger
logger = logging.getLogger(__name__)


class AnnotationBuilder:
    def __init__(
            self,
            discovery_results,
            tile_output_dir,
            tile_metadata_path,
            train_dir,
            val_dir,
            test_dir=None,
            val_split=0.2,
            test_split=0.1,
            seed=42,
            focus_label=None,
            # HACK: only for debugging, to process a single file.
            no_tile=False
    ):
        try:
            self.shapefile = discovery_results['shapefile']
            self.class_map = discovery_results['class_map']
            self.type_key = discovery_results['id_key']
            self.name_key = discovery_results['name_key']
        except KeyError as e:
            raise ValueError(f"Invalid discovery results: {e}")

        self.tile_dir = tile_output_dir
        self.metadata_path = tile_metadata_path
        self.focus_label = focus_label
        self.no_tile = no_tile

        # HACK: This is only for debugging, and is only used when no_tile=True.
        # This is used as a debugging hack to process a single tile. Set
        # no_tile=True and file_name_hack to the name of the tile you want to
        # process, and all tiles in the tile_metadata.json file will be ignored
        # except for the given file_name.
        if no_tile:
            self.file_name_hack = "Hossur_Geratti_1.png"

        self.split_manager = SplitManager(
            train_dir, val_dir, test_dir, val_split, test_split, seed)

    def _get_tile_size(self, tile_path):
        """Read image dimensions.

        @param tile_path: path to the tile image. 

        @returns tuple: the (width, height) of the tile.

        @raises ValueError: if the image can't be opened or if it's not square (when no_tile=False)
        """
        try:
            with Image.open(tile_path) as img:
                width, height = img.size
                if not self.no_tile and width != height:
                    raise ValueError(
                        f"Tile {tile_path} is not square: {width}x{height}")
                return width, height
        except Exception as e:
            raise ValueError(f"Error opening tile {tile_path}: {e}")

    def should_process_file(self, tile):
        """Determine if a file should be processed based on no_tile and file_name_hack settings.

        @param filename: The name of the file to check
        @returns bool: True if the file should be processed, False otherwise
        """
        filename = tile["filename"]
        if self.no_tile and self.file_name_hack:
            if (filename == self.file_name_hack):
                logger.info(f"Processing file: {filename}")
                return True
            else:
                logger.info(f"Skipping file: {filename}")
                return False
        # TODO(dlt/issues/37): consolidate key names with
        # tile_generator.
        return tile.get("is_valid", True)

    def run(self):
        """Build annotations and save to disk.

        @raises ValueError: if the tile metadata is corrupted. 
        @raises ValueError: if a class is not found in the class map. 
        @raises ValueError: if a class ID in the shapefile does not match the class map. 
        @raises ValueError: if a transformed annotation is out of bounds of the tile. 
        """
        with open(self.metadata_path) as f:
            tile_metadata = json.load(f)

        gdf = gpd.read_file(self.shapefile)
        gdf = GeometryValidator(gdf).fix(strict_fix=True)

        image_id = 0
        annotation_id = 0

        # Filter metadata before splitting
        # HACK: remove
        tile_metadata = [tile for tile in tile_metadata
                         if self.should_process_file(tile)]

        if not tile_metadata:
            raise ValueError("No files to process after filtering")

        self.split_manager.split_tiles(tile_metadata)

        # Each row in tile metadata:
        # {
        #   "filename": "Hulibanda_Checkdam_2_x3456_y0.png",
        #   "site": "Hulibanda_Checkdam_2",
        #   "tile_bounds": [
        #     801946.2150776262,
        #     1361264.4502956227,
        #     801971.8144030535,
        #     1361290.0500793674
        #   ],
        #   "pixel_origin": [
        #     3456,
        #     0
        #   ],
        #   "crs": "EPSG:32643"
        # }
        # And each row in the shapefile:
        # {
        #   "id": NaN,
        #   "Types": 4,
        #   "Name": "Lantana Cover",
        #   "Place": "Hulibanda Checkdam 2",
        #   "geometry": {
        #     "type": "Polygon",
        #     "coordinates": [
        #       [
        #         [801946.2150776262, 1361264.4502956227],
        #         [801946.2150776262, 1361290.0500793674],
        #         [801971.8144030535, 1361290.0500793674],
        #         [801971.8144030535, 1361264.4502956227],
        #         [801946.2150776262, 1361264.4502956227]
        #       ]
        #     ]
        #   }
        # }
        # Where the keys are: Name, id, Types, Place are custom fields in the
        # shapefile and geometry is the polygon. The geometry is auto captured
        # by tools like QGIS when the user draws a shape, the other fields are
        # user-defined through eg a popup form in QGIS.
        for tile in tqdm(tile_metadata, desc="Building annotations"):
            try:
                img_filename = tile["filename"]
                bounds = tile["tile_bounds"]
                crs = tile["crs"]
            except KeyError as e:
                raise ValueError(f"Corrupted tile metadata: {e}")

            # Convert the tile bounds, which is a list of 4 floats, to a
            # shapely polygon object
            tile_polygon = box(*bounds)
            tile_path = os.path.join(self.tile_dir, "images", img_filename)

            # Load and reproject shapefile to the CRS
            if gdf.crs.to_string() != crs:
                gdf_proj = gdf.to_crs(crs)
            else:
                gdf_proj = gdf

            #
            # Now the CRS is Geotiff (i.e. EPSG:32643)
            #

            # Note that this returns all polygons that intersect with the tile.
            # No clipping is done. "Intersects" returns true if the polygons
            # overlap, and so gdf_proj[intersects] returns all touching polygons
            # without clipping.
            intersecting = gdf_proj[gdf_proj.intersects(tile_polygon)]
            # Filter for only focus_label (eg: "Lantana Cover") polygons
            if self.focus_label:
                intersecting = intersecting[
                    intersecting[self.name_key] == self.focus_label]

            # Get tile dimensions - now returns a tuple
            width, height = self._get_tile_size(tile_path)
            image_entry = {
                "id": image_id,
                "file_name": f"images/{img_filename}",
                "width": width,
                "height": height
            }

            # This list collects all annotations for this image
            tile_annotations = []
            for _, row in intersecting.iterrows():

                # Some post intersection geometries might still be invalid. If
                # we can fix them, do so.
                if not row.geometry.is_valid:
                    fixed_geom = GeometryValidator.fix_invalid_geometry(
                        row.geometry)
                    if not fixed_geom:
                        logger.warning(
                            f"Skipping invalid geometry in tile processing")
                        continue
                    row.geometry = fixed_geom

                try:
                    clipped = row.geometry.intersection(tile_polygon)
                except Exception as e:
                    logger.error(f"Error clipping geometry: {e}")
                    continue

                if clipped.is_empty:
                    continue

                label = row[self.name_key]
                cat_id = self.class_map.get(label)
                if cat_id is None:
                    raise ValueError(f"Class {label} not found in class map")

                # Convert geospatial to pixel coords
                # Tile bounds: [801927.0155835557, 1361264.4502956227..]
                # Pixel bounds: [0, 0, 512, 512] ->> this doesn't matter here
                # What this translation does is move the tile 801927m "to the
                # right" in preperation for the next transformation that happens
                # with the intersecting clipped polygon.
                #
                # With the origin shifted, and the size of each pixel defined
                # (which the Affine.scale does), the warp.transform_geom will
                # shift the polygon to pixel coordinates relative to the tile.
                #
                # The transform happens in 3 steps:
                # 1. Translation: move the origin to the tile's origin
                # 2. Scaling: scale the polygon to the tile's size
                # 3. Intersection: compute the clipped polygon pixel coords
                # relative to this new origin.
                pixel_transform = Affine.translation(bounds[0], bounds[3]) * Affine.scale(
                    # Use width instead of tile_size
                    (bounds[2] - bounds[0]) / width,
                    # Use height instead of tile_size
                    -(bounds[3] - bounds[1]) / height
                )
                inv_transform = ~pixel_transform
                pixel_geom = shapely_affinity.affine_transform(clipped, [
                    inv_transform.a,  # a
                    inv_transform.b,  # b
                    inv_transform.d,  # d
                    inv_transform.e,  # e
                    inv_transform.c,  # xoff
                    inv_transform.f   # yoff
                ])

                # Basic polygon check
                if not pixel_geom or pixel_geom.is_empty:
                    continue

                x, y, max_x, max_y = pixel_geom.bounds
                w = max_x - x
                h = max_y - y
                logger.debug(f"Pixel-space bbox: {x}, {y}, {w}, {h}")

                # Handle both Polygon and MultiPolygon cases
                # COCO format expects segmentation to be a list of lists:
                # List[List[float]]
                # For polygons, it's: [ [...x, y, x, y...] ]
                # For multipolygons, it's: [ [...], [...] ]
                segmentation = []
                if pixel_geom.geom_type == 'MultiPolygon':
                    segmentation = [
                        [coord for pt in poly.exterior.coords for coord in pt]
                        for poly in pixel_geom.geoms
                    ]
                else:
                    segmentation = [
                        [coord for pt in pixel_geom.exterior.coords
                         for coord in pt]
                    ]

                ann = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [x, y, w, h],
                    "bbox_mode": 1,  # XYWH_ABS
                    "segmentation": segmentation,
                    "iscrowd": 0
                }

                # The transformed polygon should be within the tile bounds
                if not (0 <= x <= width and 0 <= y <= height and
                        0 <= x + w <= width and 0 <= y + h <= height):
                    raise ValueError(
                        f"Invalid annotation bounds for tile {img_filename}: "
                        f"bbox ({x}, {y}, {w}, {h}) is out of {width}x{height} bounds"
                    )

                tile_annotations.append(ann)
                annotation_id += 1

            # Add this image and all its annotations as one sample
            self.split_manager.add_sample(
                image_entry,
                tile_annotations,
                img_filename
            )
            image_id += 1

        # Batch write all images and annotations to disk
        categories = [{"id": id_val, "name": name}
                      for name, id_val in self.class_map.items()]
        self.split_manager.write(self.tile_dir, categories)


class SplitManager:
    """Manage the train/val/test splits.

    This class is responsible for splitting the tiles into train/val/test sets
    and writing the images and annotations to disk. It splits tiles based on
    the split ratios and the seed. The first N tiles are assigned to the test set, the next N tiles are assigned to the val set, and the rest are assigned to the train set.
    """

    def __init__(self, train_dir, val_dir, test_dir=None, val_split=0.2, test_split=0.1, seed=42):
        self.image_sources = []
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        # These data structures are used to record each sample. They typically
        # grow, and end up matching the train_tiles, val_tiles, test_tiles sets.
        self.train_images, self.val_images, self.test_images = [], [], []
        self.train_annotations, self.val_annotations, self.test_annotations = [], [], []

        # These data structures are used to track which tiles are assigned to
        # which split. They don't change.
        self.train_tiles, self.val_tiles, self.test_tiles = set(), set(), set()

        os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
        if test_dir:
            os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)

    def split_tiles(self, tile_metadata):
        """Split tiles into train/val/test sets."""
        random.seed(self.seed)
        random.shuffle(tile_metadata)

        total = len(tile_metadata)
        test_count = int(total * self.test_split) if self.test_dir else 0
        val_count = int((total - test_count) * self.val_split)

        self.test_tiles = set([m['filename']
                              for m in tile_metadata[:test_count]])
        self.val_tiles = set([m['filename']
                              for m in tile_metadata[test_count:test_count + val_count]])
        self.train_tiles = set([m['filename']
                                for m in tile_metadata[test_count + val_count:]])

        logger.info(
            f"Split sizes: Train: {len(self.train_tiles)}, Val: {len(self.val_tiles)}, Test: {len(self.test_tiles)}")

        return self.train_tiles, self.val_tiles, self.test_tiles

    def add_sample(self, image_entry, annotations, tile_filename):
        """Add image and its annotations to the appropriate split."""
        # Track the source image and its destination split
        if tile_filename in self.test_tiles:
            self.test_images.append(image_entry)
            self.test_annotations.extend(annotations)
            dest_split = 'test'
        elif tile_filename in self.val_tiles:
            self.val_images.append(image_entry)
            self.val_annotations.extend(annotations)
            dest_split = 'val'
        else:
            self.train_images.append(image_entry)
            self.train_annotations.extend(annotations)
            dest_split = 'train'

        self.image_sources.append((tile_filename, dest_split))

    def write_images(self, source_dir):
        """Write all images to their respective split directories in batch."""
        logger.info("Writing images to split directories...")

        # Get all filenames from image_sources
        processed_files = {filename for filename, _ in self.image_sources}

        # Get all files that should have been processed
        expected_files = self.train_tiles | self.val_tiles | self.test_tiles

        # Check for missing files
        missing_files = expected_files - processed_files
        if missing_files:
            raise RuntimeError(
                f"Found {len(missing_files)} tiles that were split but never processed: "
                f"{sorted(list(missing_files))[:5]}{'...' if len(missing_files) > 5 else ''}"
            )

        split_dirs = {
            'train': self.train_dir,
            'val': self.val_dir,
            'test': self.test_dir
        }

        for tile_filename, split in tqdm(self.image_sources, desc="Copying images"):
            src_path = os.path.join(source_dir, "images", tile_filename)
            dst_path = os.path.join(split_dirs[split], "images", tile_filename)

            try:
                shutil.copy(src_path, dst_path)
            except (IOError, OSError) as e:
                raise RuntimeError(
                    f"Failed to copy image {tile_filename} to {split} split: {str(e)}")

    def write_annotations(self, categories):
        """Write COCO-format annotations for all splits."""
        logger.info("Writing annotations...")

        def build_coco(images, annotations):

            # Example annotations:
            # {
            #     "images": [
            #         {
            #             "id": 0,
            #             "file_name": "images/image1.png",
            #             "width": 512,
            #             "height": 512
            #         },
            #         {
            #             "id": 1,
            #             "file_name": "images/image2.png",
            #             "width": 512,
            #             "height": 512
            #         }
            #     ],
            #     "annotations": [
            #         {
            #             "id": 0,
            #             "image_id": 0,
            #             "category_id": 1,
            #             "bbox": [100.0, 150.0, 80.0, 60.0],
            #             "bbox_mode": 1,
            #             "segmentation": [[100.0, 150.0, 180.0, 150.0, 180.0, 210.0, 100.0, 210.0]],
            #             "iscrowd": 0
            #         },
            #         {
            #             "id": 1,
            #             "image_id": 1,
            #             "category_id": 2,
            #             "bbox": [200.0, 120.0, 50.0, 50.0],
            #             "bbox_mode": 1,
            #             "segmentation": [[200.0, 120.0, 250.0, 120.0, 250.0, 170.0, 200.0, 170.0]],
            #             "iscrowd": 0
            #         }
            #     ],
            #     "categories": [
            #         {
            #             "id": 1,
            #             "name": "Lantana"
            #         },
            #         {
            #             "id": 2,
            #             "name": "Senna"
            #         }
            #     ]
            # }
            return {
                "images": images,
                "annotations": annotations,
                "categories": categories
            }

        try:
            with open(os.path.join(self.train_dir, "annotations.json"), "w") as f:
                json.dump(build_coco(self.train_images,
                          self.train_annotations), f)

            with open(os.path.join(self.val_dir, "annotations.json"), "w") as f:
                json.dump(build_coco(self.val_images, self.val_annotations), f)

            if self.test_dir:
                with open(os.path.join(self.test_dir, "annotations.json"), "w") as f:
                    json.dump(build_coco(self.test_images,
                              self.test_annotations), f)

        except IOError as e:
            raise RuntimeError(f"Failed to write annotations: {str(e)}")

    def write(self, source_dir, categories):
        """Write both images and annotations."""
        try:
            self.write_images(source_dir)
            self.write_annotations(categories)
        except Exception as e:
            raise RuntimeError(f"Failed to write dataset: {str(e)}")
