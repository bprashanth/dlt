# Classes, Categories and Indices 


## Validating COCO

Run 
```
$ python hack/validate_coco.py --coco data/train/annotations.json --log_level DEBUG
```

## Overview of categories 

See [this](https://github.com/bprashanth/dlt/issues/15) issue for more context.
Tl;dr is we need a consiste way to ensure that indices and classes match up -
right now we just trust the input dataset. 

In discovery we read the categories out of the shp file using the supplied column name (`--name_key`)
```
# Create initial name->id mapping from unique combinations
class_map = dict(zip(gdf[self.name_key], gdf[self.id_key]))

# Remove NaN/None entries
class_map = {k: v for k, v in class_map.items() if pd.notna(k)
	     and pd.notna(v)}
```
That gives us a transformation of 
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
To 
```
"class_map": {
    "Open Space": 0,
    "Senna": 1,
    "Bamboo": 2,
    "Water Bodies": 3,
    "Lantana Cover": 4,
    "Trees": 5,
    "Agriculture Land": 7,
    "Coconut Tree": 8,
    "House": 9,
    "Chromolaena": 6
  },
```

In `annotation_builder` this is translated into a `category id`
```
label = row[self.name_key]
cat_id = self.class_map.get(label)
```

While at this point it makes no difference whether we use `row[self.id_key]` or `cat_id` as computed above, it could in the future. This `cat_id` is added to the annotations, and appended to the end of the coco file as `categories`
```
ann = {
    "id": annotation_id,
    "image_id": image_id,
    "category_id": cat_id,
    "bbox": [x, y, w, h],
    "bbox_mode": 1,  # XYWH_ABS
    "segmentation": [segmentation],
    "iscrowd": 0
}
...
categories = [{"id": id_val, "name": name}
	      for name, id_val in self.class_map.items()]
```
which should give us the exact same ids at the end of annotations.json
```
$ cat ./data/train/annotations.json | jq | tail -50
  "categories": [
    {
      "id": 0,
      "name": "Open Space"
    },
    {
      "id": 1,
      "name": "Senna"
    },
    {
      "id": 2,
      "name": "Bamboo"
    },
    {
      "id": 3,
      "name": "Water Bodies"
    },
    {
      "id": 4,
      "name": "Lantana Cover"
    },
    {
      "id": 5,
      "name": "Trees"
    },
    {
      "id": 7,
      "name": "Agriculture Land"
    },
    {
      "id": 8,
      "name": "Coconut Tree"
    },
    {
      "id": 9,
      "name": "House"
    },
    {
      "id": 6,
      "name": "Chromolaena"
    }
  ]
}
```

Then, during training we supply the categories in 2 places: 
1. Metadata, as a list of classes which are computed as such
```
self.categories = sorted(
    self.data['categories'], key=lambda x: x['id'])
self.class_names = [cat['name'] for cat in self.categories]
...
MetadataCatalog.get("train_dataset").set(thing_classes=class_names)
```

2. As part of `register`, in the annotations 
```
register_coco_instances(
    "train_dataset", {}, self.train_coco_path,
    os.path.join(self.train_dir, "images")
)
```

So if the metadata class list is 
```
["Open Space", "Senna", "Bamboo"]
```
We need the following in annotations.json's categories 
```
"categories": [
  {"id": 0, "name": "Open Space"},
  {"id": 1, "name": "Senna"},
  {"id": 2, "name": "Bamboo"}
]
```
