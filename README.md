# dlt

Drone Lantana Detectors

## Running detectors 

Eg to run the samclip directory
```
$ cd samclip
$ docker run -it --rm \
  -v $(pwd):/app \
  -w /app \
  sam-clip \
  bash
```
