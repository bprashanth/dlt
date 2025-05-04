# Debugging tips 

Run with just 1 class label and one file, no tiling 
```
$ python3 main.py --root_dir ~/rtmp/data/shola/data/ --log_level INFO --tile_output_dir ./data/tiles --val_dir ./data/val --train_dir ./data/train --test_dir ./data/test --pipeline_config ./pipeline_config.json --no_tile --focus_label "Water Bodies"

$ python3 ./hack/shp_on_tiff.py --shp path/to/shp --tiff path/to/tiff

$ python3 ./hack/coco_on_png.py --png ./data/train/images/Hossur_Geratti_1.png --coco ./data/train/annotations.json
```
And compare the two images 

Available options are (run `hack/type_name_combos.py` to find)
```
Unique Types-Name combinations:
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

To debug the data frame in python
```
>>> gdf.crs
<Geographic 2D CRS: EPSG:4326>
Name: WGS 84
Axis Info [ellipsoidal]:
- Lat[north]: Geodetic latitude (degree)
- Lon[east]: Geodetic longitude (degree)
Area of Use:
- name: World.
- bounds: (-180.0, -90.0, 180.0, 90.0)
Datum: World Geodetic System 1984 ensemble
- Ellipsoid: WGS 84
- Prime Meridian: Greenwich

>>> print(gdf['Name'].unique())
['Open Space' 'Senna' 'Bamboo' 'Water Bodies' 'Lantana Cover' 'Trees'
 'Agriculture Land' 'Coconut Tree' 'House' 'Chromolaena']
>>> print(gdf['Place'].unique())
['Gavihalla' 'Hulibanda Checkdam 2' 'Hossur Geratti 1' 'Hossur Geratti 2'
 'Hulibanda Checkdam 1' 'Hulibanda Cleared plot 3'
 'Hulibanda Cleared plot 1' 'Jawal Lakshmipura 1' 'Jawal Lakshmipura 2'
 'Marapallam' 'Moyar' 'Wayanad' 'Tholuvu Betta']

>>> filtered_gdf = gdf[(gdf['Name'] == 'Water Bodies') & (gdf['Place'] == 'Hossur Geratti 1')]
>>> filtered_gdf
     id  Types          Name             Place                                           geometry
430 NaN      3  Water Bodies  Hossur Geratti 1  POLYGON ((77.77753 12.28721, 77.7775 12.28726,...
>>> filtered_gdf.geometry
430    POLYGON ((77.77753 12.28721, 77.7775 12.28726,...
Name: geometry, dtype: geometry
>>> coords = filtered_gdf.geometry.iloc[0].exterior.coords.xy
>>> coords[0]
...
>>> coords[1]
```
To skip stages of the pipeline, modify `pipeline_config.json`
```
{
    "skip_validation": false,
    "skip_tiling": false,
    "skip_annotation": false
}

```

To only generate labels for 1 class 
```
$ python3 main.py --root_dir ~/rtmp/data/shola/data/ --log_level INFO --tile_output_dir ./data/tiles --val_dir ./data/val --train_dir ./data/train --test_dir ./data/test --pipeline_config ./pipeline_config.json --tile_size 2048 --focus_label "Lantana Cover"
```
