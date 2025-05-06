# Debugging tips 

Run with just 1 class label and one file, no tiling 
```
$ python3 main.py --root_dir ~/rtmp/data/shola/data/ --log_level INFO --tile_output_dir ./data/tiles --val_dir ./data/val --train_dir ./data/train --test_dir ./data/test --pipeline_config ./pipeline_config.json --no_tile --focus_label "Water Bodies"

$ python3 ./hack/shp_on_tiff.py --shp path/to/shp --tiff path/to/tiff

$ python3 ./hack/coco_on_png.py --png ./data/train/images/Hossur_Geratti_1.png --coco ./data/train/annotations.json

$ python3 ./hack/tiles_on_tiff.py --tiff ~/rtmp/data/shola/data/Master_Labelled/Hossur\ Geratti\ 1/Hossur_Geratti_1.tiff --tile_metadata ./data/tiles/tiles_metadata.json --output output.png --shp ~/rtmp/data/shola/data/Labels_Polygons_All.shp
```
And compare all these images 

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

## Skippping stages in the pipeline 

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

## Errors in annotation 

During model training if you see a very low number of classes in something you know to be present in the annotations, most likely you have messed up the coco format. 
Eg, if you know there is a distribution of labels but the category map shows something like this during model training 
```console 
2025-05-05 11:14:10,459 - detectron2.data.build - INFO - Distribution of instances among all 10 categories:
|   category   | #instances   |   category    | #instances   |   category   | #instances   |
|:------------:|:-------------|:-------------:|:-------------|:------------:|:-------------|
|  Open Space  | 0            |     Senna     | 0            |    Bamboo    | 0            |
| Water Bodies | 0            | Lantana Cover | 1443         |    Trees     | 0            |
| Chromolaena  | 0            | Agriculture.. | 0            | Coconut Tree | 0            |
|    House     | 0            |               |              |              |              |
|    total     | 1443         |               |              |              |              |

```

## Data Frame debugging 

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

### Null and corrupt geometries 

```
>>> gdf[gdf['geometry'].isnull()]
      id  Types           Name             Place geometry
702  NaN      4  Lantana Cover  Hossur Geratti 1     None
3343 NaN      5          Trees     Tholuvu Betta     None

>>> gdf[~gdf.geometry.is_valid]
      id  Types           Name                 Place                                           geometry
27   NaN      0     Open Space             Gavihalla  POLYGON ((76.64797 11.58985, 76.64796 11.58984...
177  NaN      0     Open Space  Hulibanda Checkdam 2  POLYGON ((77.77555 12.2963, 77.77555 12.29631,...
253  NaN      4  Lantana Cover  Hulibanda Checkdam 2  POLYGON ((77.77556 12.2993, 77.77552 12.2993, ...
269  NaN      4  Lantana Cover  Hulibanda Checkdam 2  POLYGON ((77.77577 12.29935, 77.77579 12.29936...
338  NaN      4  Lantana Cover  Hulibanda Checkdam 2  POLYGON ((77.77647 12.29837, 77.77645 12.29836...
...   ..    ...            ...                   ...                                                ...
2612 NaN      5          Trees         Tholuvu Betta  POLYGON ((77.84265 12.37129, 77.84263 12.37131...
2994 NaN      5          Trees         Tholuvu Betta  POLYGON ((77.84299 12.36957, 77.84296 12.36962...
2998 NaN      5          Trees         Tholuvu Betta  POLYGON ((77.84254 12.37005, 77.84251 12.37005...
3067 NaN      5          Trees         Tholuvu Betta  POLYGON ((77.84265 12.37129, 77.84263 12.37131...
3343 NaN      5          Trees         Tholuvu Betta                                               None

[90 rows x 5 columns]
```

