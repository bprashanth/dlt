# Tiling 

## Handlign overlap

Overlap handling happens in two places
1. Between tile generator and tile stitcher 
2. In metrics 

## Tile stitching

The first is pretty automatic. 
This is how we tile. 
```
step = self.tile_size - self.overlap
```
This means if `tile_size = 512` and `overlap = 128`, then each tile starts 384 pixels after the previous one, causing a 128-pixel overlap.
But we also record in `tile_metadata.json`
```
"pixel_origin": [x, y]
```
And this origin point tells the stitcher where to place the tile. 
The stitcher doesn't need to know the overlap because it can simply drawn an empty canvas, and place tiles in their origin points, and that would take care of the overlap. 

## Metrics 

In metrics, this is a little more complicated. This is the process we follow 
1. Create a large canvas to represent the full map
2. For each polygon from a given tile (in the coco predictions json) we 
	a. Shift it using `pixel_origin`
	b. Add the polygon to the global canvas as a class mask 
	c. Rasterize only once, meaning if a pixel is marked as class A leave it as such 

What this essentially means in pseudo code is 
1. Create a 2D map of (H,W) of full map 
2. For each tile's polygons found in the coco json, shift it using `pixel_origin`
	- the list of pixel values in the coco json is local to the 2048 tile
3. For each class maintain a binary mark, add to it without overlap
4. Count total pixels per class in global mask

Eg the class masks would look like 
```python 
masks = {
  "Lantana Cover": np.zeros((max_y, max_x), dtype=bool),
  "Bare Soil": np.zeros((max_y, max_x), dtype=bool),
  ...
}
```

