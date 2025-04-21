# Detectron2

Drone Lantana Detectors based on Detectron2

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
$ docker run -d --rm --net host -v $(pwd):/app detectron2:0.1 python train.py > training.log 2>&1
```
The checkpoints are saved  to `output/checkpoints`

## Inference 

If `train_data` is set to None/empty, the program runs inference against the images in `--inference_data` using the weights in `--weights_path`. 

```
$ python ./train.py --train_data "" --inference_data ./data/val/ --weights_path ./output/checkpoints/output/model_final.pth
```

The output is saved to `output/images`

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
Once the validation is complete, you must manually copy the  data into the `train`, `val` and `test` dirs before training the model. 

## Assets

* SAMCLIP
	- Checkpoint: VIT-B SAM [model](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
	- Source: [SAM](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) from meta
