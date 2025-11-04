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
$ docker build -t sam-clip:0.2 .
```

Eg to run the samclip directory
```console 
$ cd samclip
$ docker run -it --rm \
  --net host \
  -v $(pwd):/app \
  -w /app \
  sam-clip:0.2 \
  bash
$ rm -rf ./output/* && python sam_clip.py --image lantana.jpg --output_dir ./output/ --model {maskrcnn, samclip, groundingdino}
# OR, by specifying the prompt 
$ python sam_clip.py --image ./input_images/IMG-20250924-WA0001.jpg --clip-text_prompt "a hand filled survey form" 
```


## Assets

* SAMCLIP
	- Checkpoint: VIT-B SAM [model](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
	- Source: [SAM](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) from meta
