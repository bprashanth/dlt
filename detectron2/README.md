# Detectron2

Drone Lantana Detectors based on Detectron2

For a quickstart and a brief overview of the pipeline, continue reading. For everything else, see [docs](./docs/). 

## Pipeline 

```console 
     1. Discovery 
	| - "discover" all files of interest. What to ignore. What to train on.
	| - Format into internal data structure for consistency. 
	|
     2. Validation
	| - Are the polygons in the shp file intersecting the geotiffs?
	| - Are there Null or corrupt polygons, are the fixable? 
	| - Are the CRS' and datums compatible?
	|
     3. TileGenerator
	| - Chunk geotiff into w=(512-2048), h=(512-2048) squares 
	| - Apply sliding window 
	| - Handle edge effects
	| - Write out intermediate tile_metadata.json with tile boundaries 
	|
     4. Pre-processing 
	| ??? (normalize, data augmentation etc) 
	| 
     5. AnnotationBuilder
	| - Intersect shp polygons with tile boundaries 
	| - Clip where necessary 
	| - Translate to pixel coordinates 
	| - Write in coco format 
	| - Compute and manage reproducible splits 
	|
     6. Manual verification
	| - Plot coco on base tiles 
	| - Plot shp annotations on tiff
	| - Visually sanity check differences 
	| 
     7. Training (docker/remote server) 
	| - Validate coco json, are the classes and indices aligned, are there malformed annotations etc
	| - Register datasets with model
	| - Train model to output checkpoints
	| 
     8. InferenceRunner (docker/remote server)
	| 
	| Batch (file) mode 
	| - Load metadata from coco 
	| - Load weights from training 
	| - Predict polygons 
	| - Draw polygons on input image and write output png 
	| - Write prediction coco.json 
	|
	| UI mode 
	| - Start the UI server 
	| - Wait for user to choose weights, image, confidence, classes, ground truth
	| - Predict polygons 
	| - Draw polygons on input image and display
	| - If ground truth annotations specified, display for comparison
	|
	User chooses Gradio Mode 
	 \--> Launch Docker --> Run Gradio server 
          \--> User uploads image, chooses weights + threshold + classes --> InferenceRunner is invoked per image 
	---|
	|
	|
     9. Test scoring 
	| - Compare test coco w/ prediction coco for test scores 
	| - IOU: GT (test annotations) 
	|	 TP (predicted polygon matches GT > threshold + correct class) 
	|        FP (predicted polygon doesn't overlap GT)
	|        FN (a GT polygon with no matching prediction)
	| - Diminishing returns: how much data do we need? 
	| 	Loss curves
	| 	Ablation
	|
     10. Post-processing
	| - Per (site, category, tile) metrics 
	| ??? (stitch tiles, generate histograms/heatmaps)
```

- Steps 1-4: Discovery, validation, tiling and COCO annotation transformation

Set `--pipeline_config` to 
```json
{
    "skip_validation": false,
    "skip_tiling": false,
    "skip_annotation": false,
    "skip_training": true,
    "skip_inference": true
}
```
To prep data with a 70/20/10 split and 2048x2048 pixel tiles, only over the "Lantana Cover" class
```
$ python3 main.py --root_dir ~/rtmp/data/shola/data/ --log_level INFO --tile_output_dir ./data/tiles --val_dir ./data/val --train_dir ./data/train --test_dir ./data/test --pipeline_config ./pipeline_config.json --tile_size 2048 --focus_label "Lantana Cover"
```

Or, just remove `--focus_label` to train with all known classes. 
Make sure you visually inspect the annotations are correct before moving to the next stage (see [this](docs/data.md) for instructions).

- Step 5: Train

Set `pipeline_config` to
```json
{
    "skip_validation": true,
    "skip_tiling": true,
    "skip_annotation": true,
    "skip_training": false,
    "skip_inference": true
}
```
Then run
```
$ python3 main.py --root_dir ~/rtmp/data/shola/data/ --tile_output_dir ./data/tiles --val_dir ./data/val --train_dir ./data/train --test_dir ./data/test --pipeline_config ./pipeline_config.json --tile_size 2048 --checkpoint_output_dir ./checkpoints/all --training_image detectron2:1.3 --log_level INFO
```

- Step 6: Inference

Set `pipeline_config` to
```json
{
    "skip_validation": true,
    "skip_tiling": true,
    "skip_annotation": true,
    "skip_training": true,
    "skip_inference": false
}
```
There are 2 modes of inference: 
1. Single image mode 
2. Through the UI
3. Batch mode (unimplemented)

Single image mode 
```
$ source venv/bin/activate && ./run_inference.sh -image ./data/test/images/Hulibanda_Cleared_plot_1_x7467_y3840.png -output_dir ./inference -weights ./checkpoints/all/model_final.pth 
```
This will generate 2 images in `./inference`
1. The base png + predictions 
2. The base png + annotations taken from `--test_dir`/annotations.json

Or, if you would prefer the gradio interface (same command with `-gradio`)
```
$ source venv/bin/activate && ./run_inference.sh -image ./data/test/images/Hulibanda_Cleared_plot_1_x7467_y3840.png -output_dir ./inference -weights ./checkpoints/all/model_final.pth -gradio
```
And modify the confidence level and selected classes appropriately. 

## Data "Discovery" 

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

