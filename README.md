# DLT

Drone Lantana Detectors

## Running detectors 

Directories

```console 
samclip/
├── segment-anything/   ← pulled from upstream, clean and trackable
├── Dockerfile
├── requirements.txt
├── app/
│   └── run_sam_clip.py
```

Build 
```console 
$ docker build -t sam-clip .
```

Eg to run the samclip directory
```console 
$ cd samclip
$ docker run -it --rm \
  -v $(pwd):/app \
  -w /app \
  sam-clip \
  bash
$ python app/run_sam_clip.py
```


## Assets

* SAMCLIP
	- Checkpoint: VIT-B SAM [model](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
	- Source: [SAM](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) from meta
