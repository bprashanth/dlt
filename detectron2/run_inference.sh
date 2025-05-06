#!/bin/bash

# Check if both arguments are provided
if [ $# -ne 4 ]; then
    echo "Usage: $0 -image <image_path> -output_dir <output_directory>"
    exit 1
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -image)
            IMAGE="$2"
            shift 2
            ;;
        -output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check if required arguments are set
if [ -z "$IMAGE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Both -image and -output_dir must be specified"
    exit 1
fi

# Extract just the filename from the full path
file_name=$(basename "${IMAGE}")

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the inference commands
python3 main.py --root_dir ~/rtmp/data/shola/data/ \
    --tile_output_dir ./data/tiles \
    --val_dir ./data/val \
    --train_dir ./data/train \
    --test_dir ./data/test \
    --pipeline_config ./pipeline_config.json \
    --tile_size 2048 \
    --checkpoint_output_dir ./checkpoints/all \
    --training_image detectron2:1.0 \
    --inference_weights_path ./checkpoints/lantana/model_final.pth \
    --inference_test_image "${IMAGE}" \
    --inference_output_dir "${OUTPUT_DIR}" \
    --inference_image detectron2:1.0 \
    --log_level INFO

python3 ./hack/coco_on_png.py --png "${IMAGE}" \
    --coco ./data/test/annotations.json \
    --output "${OUTPUT_DIR}/${file_name}" 
