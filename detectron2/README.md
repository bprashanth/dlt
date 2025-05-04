# Detectron2

Drone Lantana Detectors based on Detectron2

## Quickstart

- Discovery: discover the site tiff files and the main shp file
- Validation: validate the polygons intersect between the tiffs and the shp file
- Tiling: split up the tiff into tiled 'pngs'
- Annotation Transformation: intersect the bounds of each tile with the main shp file and generate COCO annotations, clipping polygons where necessary
- Training: fine-tune detectron2 on the COCO annotations
- Infernce: run inference on new images

- Steps 1-4: Discovery, validation, tiling and COCO annotation transformation

To prep data with a 70/20/10 split and 2048x2048 pixel tiles, only over the "Lantana Cover" class
```
$ python3 main.py --root_dir ~/rtmp/data/shola/data/ --log_level INFO --tile_output_dir ./data/tiles --val_dir ./data/val --train_dir ./data/train --test_dir ./data/test --pipeline_config ./pipeline_config.json --tile_size 2048 --focus_label "Lantana Cover"
```

- Step 5: Train

```
docker run -d --rm --net host -v $(pwd):/app detectron2:0.1
```

- Step 6: Inference

```
docker run -it --rm --net host -v $(pwd):/app --entrypoint /bin/bash detectron2:0.1
$ python ./main.py --train_data "" --inference_data ./data/val/ --weights_path ./output/checkpoints/output/model_final.pth
```

## Data Processing

The scripts in this directory will help assess and reformat datasets.

```
$ source venv/bin/activate
$ python3 ./data_manager.py --root_dir ...
```

The goal is to transform the data from eg

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

Where the annotations are in `COCO` json format.
Once the validation is complete, you must manually copy the data into the `train`, `val` and `test` dirs before training the model.
The validation script will also print out a list of classes, which you currently need to replace in the `main.py` file.

## Training

The docker container is setup for training. If the data is laid out in the following format

```
root_dir/
├── val
├── train/
│   └── annotations.json
│   └── imag1.png
│
├── test/
│   └── annotations.json
│   └── imag1.png
...
```

simply running the docker will run training.

```console
$ docker build -t detectron2:0.1 .
$ docker run -d --rm --net host -v $(pwd):/app detectron2:0.1
```

To save the logs to file

```
$ docker run -d --rm --net host -v $(pwd):/app detectron2:0.1 python main.py > training.log 2>&1
```

The checkpoints are saved to `output/checkpoints`

## Inference

If `train_data` is set to None/empty, the program runs inference against the images in `--inference_data` using the weights in `--weights_path`.

```
$ docker run -it --rm --net host -v $(pwd):/app --entrypoint /bin/bash detectron2:0.1
$ python ./main.py --train_data "" --inference_data ./data/val/ --weights_path ./output/checkpoints/output/model_final.pth
```

The output is saved to `output/images`

## Assets

- SAMCLIP
  - Checkpoint: VIT-B SAM [model](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
  - Source: [SAM](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) from meta
