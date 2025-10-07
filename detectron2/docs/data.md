# Data Standards 

## Input standards 

The goal of the `discovery -> validation -> tiling` parts of the pipelin is to transform the data from eg
```
root_dir/
├── Labels_Polygons_All.shp (and friends)
├── Gavihalla/
│   └── Gavihalla.tiff
├── Hosur Gerati/
│   └── Hosur Gerati.tiff
...
```

Into

```
app/
├── test
├── train/
│   └── annotations.json
│   └── imag1.png
│
├── val/
│   └── annotations.json
│   └── imag1.png
...
```
This requires a few standards.

1. __Directory layout__: The directories should match the description above. 
2. __Schema__: The `shp` files should match this schema (if it doesn't we will insert blanks, but that could end up corrupting the dataset). 
```
Driver: ESRI Shapefile
CRS: EPSG:4326
Geometry Type: Polygon

Attribute Fields (Schema):
  id: int:10
  Types: int:10
  Name: str:100
  Place: str:100

Attributes of the First Feature:
  id: None
  Types: 0
  Name: Open Space
  Place: Gavihalla
```
3. __Categories__: 

As much as possible, match these categories 
```
 Types             Name
     0       Open Space
     1            Senna
     2           Bamboo
     3     Water Bodies
     4    Lantana Cover
     5            Trees
     6      Chromolaena
     7 Agriculture Land
     8     Coconut Tree
     9            House
```
If you have a new category, add it at the end (i.e index: 10). 
If you only have a subset of these categories, use the same indices (e.g. always keep `Types: 4 == Name: Lantana Cover`).
See [docs/categories.md](./categories.md) for more details.

## Exploring the data 

* Viewing the annotations laid over the tiff
```
$ python3 ./hack/shp_on_tiff.py --tiff /home/desinotorious/rtmp/data/shola/data/Master_Labelled/Hossur\ Geratti\ 1/Hossur_Geratti_1.tiff --shp /home/desinotorious/rtmp/data/shola/data/Labels_Polygons_All.shp
```
* To see if these annotations intersect with a tile in the final coco annotation run 
```
$ python3 ./hack/tiles_on_tiff.py --tiff ~/rtmp/data/shola/data/Master_Labelled/Hossur\ Geratti\ 1/Hossur_Geratti_1.tiff --tile_metadata ./data/tiles/tiles_metadata.json --output output.png --shp ~/rtmp/data/shola/data/Labels_Polygons_All.shp 
```
* To draw the coco annotations on the png 
```
$ python3 ./hack/coco_on_png.py --png ./data/train/images/Hossur_Geratti_1.png --coco ./data/train/annotations.json
```
* Finding class name distributions: Change the path in the shp variable in the `type_name_combos.py` script and run 
```
$ python3 ./hack/type_name_combos.py
```
See debugging [doc](./docs/debug.md) for more tips 

## Data Partitioning 

* Data partitioning happens in many stages, but essentially 2 major
  segments: 
	1. Tiling the orthomosaic, and writing the borders of the tile
(say it's a 512x512 tile) into an intermediate json file in geospatial
coords 
	2. Clipping the polygons that intersect each tile's boundaries, and writing the clipped polygon + class into coco annotations 

### Tile Borders

* We move a sliding window across the image (orthomosaic) to chunk up the map and polygons (Eg: 512x512 with 20% overlap)
* Convert pixel window to geospatial bounds 

### Coordinate mapping 

* The polygons in the shapefile are in geospatial coords 
* Detectron2 expectes pixel coords relative to the window 
* Affine transform is applied to map coords 

### Polygon clipping 

When we snip the tiles, we need to store the window origin (in the CRS) and use it to intersect + compute the pixel offsets for the final annotations. 

* If the polygon in the shp file exceeds the window boundary, it's clipped 
* If the clipped segment is too small, discard it 

### Overlap

Tiling is not necessary, it just mitigates edge effects. 

* Overlap is a mitigation strategy for jagged clipping of annotated data 
* The overlapping tiles don't need to be ordered for the image to learn 
* We need to pad the last tile to maintain consistency because d2 expects uniform inputs 


## Regular vs irregular images 

* Orthomosaic clipped into 512x512 rectangles 
* Helps with standardized feature extraction 
* Frameworks might resize on the fly, which distorts data 


## Masks vs background images 

* Ideally we want multi-class content + background 
* Still useful if we overlay the mask on varied backgrounds 
* Are there qgis tools to auto generate tiles? 
* Can we tile and then annotate instead of the other way around? 

## Augmentation 

* Overlapping window methods are a basic form of augmentation
* Also a technique to split the mosaic to squares 
	- the position of the patches within tiles changes 
	- flips and rotations help too 
* Bad if overlap is too high (>=50%)

## Handling Edge effects

* Clipping polygons 
	- sliding window across the orthomosaic
	- polygons are fragmented 
	- if an image has a very small polygon, discard it 
	- what about images with no polygons? 

## File layout

The only reason to keep folders per site is to train on 2 sites and validate on one, unknown site. This is the preferred layout for detectron2 (flatter structure)

```
dataset/
├── images/
│   ├── tile_0001.png
│   ├── tile_0002.png
│   ├── ...
├── annotations.json
```

## Normalization 

Current situation

1. Images from same drone capture 
2. Likely consistent lighting conditions, altitude, camera settings 
3. Similar color distributions and exposure levels 

So we don't normalize 

## Downscaling 

The output orthomosaics are simply too large to transmit over the wire and display in the frontend so we downscale them by a _lot_. See `tile_stitcher.py` for details. To run the downscaling standalone: 
```
$ python3 downscale_image.py --image ../inference/ inference_Jawal_Lakshmipura_1_x1536_y3072.png --output downscaled.png --scale_factor 0.1
```

## Generating output excels 

You can convert the `output_tile_metadata.json` (output of the `MapOffloader`) to excel as follows 
```shell
$ python3 ./hack/excelify_output.py --input inference/signed_tile_metadata.json --output inference/signed_tile_metadata.xlsx
```
