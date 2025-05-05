# Notes on training

In Trainer we re-parse COCO JSON and generate Detectron2-internal format via `get_dataset_dicts`. This is a bit of a hack, but it's necessary because Detectron2 expects a specific format for training (fields like `file_name`, `image_id`, `height`, `width`, `annotations`). Filed dlt/issues/26.

## Validation of COCO input

In various places, we need classes, class_ids etc. All this should be read directly from the COCO JSON. As long as we validate that COCO is compliant, there shouldn't be problems with eg offsetting indices in training code etc.

Moreover `register_metadata` needs to be consistent with `register_dataset`. The former is only used for inference, the latter for training, however if they're called with different class ids they don't predict the right thing.

Validating and re-using COCO as the internal format helps here too. This indicates that we might need a standalone library for COCO-Detectron2 stages. Something that:

1. Validates COCO JSON
2. Generates Detectron2-internal format
3. Returns class ids, class names, etc

## Decoupling inference

1. Current run_inference uses hadcoded `train_dataset` to fetch `thing_classes` via `MetadataCatalog`. We need to pass `thing_classes` explicitly.
2. We can also separate metadata setup from visualisation logic through a `InferenceRunner` class.
