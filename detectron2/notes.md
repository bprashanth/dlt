# Segmenting cell platlets 

## Faster RCNN vs MaskRCNN

* FasterRCNN: Exact boundary is irrelevant, just the class and
  approx location 
* MaskRCNN: individual weeds, tumors for removal


## Detectron2 

2 stage approach
* RPN to generate "candidate regions" (may be an obj)  
* Object detection: maskrcnn, fast rcnn 

Step 1: RPN
* CNN that takes images -> candidate regions 
* Score for each region saying how likely that region is 
* Predicts a bounding box for each region 

Step 2: MaskRCNN for either instance/object or panoptic segmentation 


## Data 

* 5 images for training (1, 11, 21, 31, 41) 
* 2 imaegs for validation (5, 35)
* All images in test dataset

Convert images to 8 bit for better contrast 
```
$ sudo apt-get install imagemagick
$ magick input.tiff -depth 8 -auto-level output_%03d.png
```


## Appendix 

* [source](https://www.youtube.com/watch?v=JIPbilHxFbI&t=1160s)
* Raw [data](https://leapmanlab.github.io/dense-cell/)
* Direct link to [data](https://www.dropbox.com/scl/fi/fg6y2dafj7116vfuucgcm/platelet_data_1219.zip?rlkey=afvwo6f6oiab1pc76peirjxzd&e=1)
