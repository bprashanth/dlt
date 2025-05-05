# Data explorations 

* Viewing the annotations laid over the tiff
```
$ python3 ./hack/shp_on_tiff.py --tiff /home/desinotorious/rtmp/data/shola/data/Master_Labelled/Hossur\ Geratti\ 1/Hossur_Geratti_1.tiff --shp /home/desinotorious/rtmp/data/shola/data/Labels_Polygons_All.sh
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




