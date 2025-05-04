# Coordinate Referenc Systems

* The original shapefile is in `EPSG:4326 (WGS 84)` -> in (lon, lat)
* The geotiff is in `EPSG:32643` -> UTM Zone 43N (in meters, with origin bottom-left) 
* The shapefile is reprojected to match the TIFF CRS (the tile boundaries) 
* Then we map a CRS coordinate to a pixel coordinate within a tile 

In geospatial coordinates, Y-axis increases as you go up
```
Geospatial (CRS: EPSG:32643)

^ y (north)
|
|
+---------> x (east)
(0,0) bottom-left
```

In image/pixel grids y axis increases as you go down 
```
Pixel grid (Image space)

(0,0) top-left
+---------> x (right)
|
|
v y (down)
```
So when we go from geospatial to pixel coordinates we need to do 3 things 
1. Move the origin to the edge of the tile box 
2. Scale the pixel size to meters 
3. Flip the y axis 
