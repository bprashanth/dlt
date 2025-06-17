#!/bin/bash

# Check if minimum required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 -image <image_path> -output_dir <output_directory> -weights <weights_path> [-confidence <threshold> -gradio -min -classes <class_list> -pipeline_config <config_path>]"
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
        -weights)
            WEIGHTS="$2"
            shift 2
            ;;
        -confidence)
            CONFIDENCE="$2"
            shift 2
            ;;
        -gradio)
            GRADIO_MODE="true"
            shift 1
            ;;
        -min)
            MINIMAL_VIS="true"
            shift 1
            ;;
        -classes)
            CLASSES="$2"
            shift 2
            ;;
        -pipeline_config)
            PIPELINE_CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Set default confidence if not specified
# TODO(prashanth@): tune this higher, eg 0.3
CONFIDENCE=${CONFIDENCE:-0.1}

# Set default pipeline config if not specified
PIPELINE_CONFIG=${PIPELINE_CONFIG:-./pipeline_config.json}

# Check if required arguments are set
if [ -z "$OUTPUT_DIR" ] || [ -z "$WEIGHTS" ]; then
    echo "All arguments (-image, -output_dir, and -weights) must be specified"
    exit 1
fi

# Extract just the filename from the full path
file_name=$(basename "${IMAGE}")

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Skip COCO annotation overlay in minimal mode
if [ -z "$MINIMAL_VIS" ] && [ -f "${IMAGE}" ]; then
    python3 ./hack/coco_on_png.py --png "${IMAGE}" \
        --coco ./data/test/annotations.json \
        --output "${OUTPUT_DIR}/${file_name}" 
elif [ -n "$MINIMAL_VIS" ]; then
    echo "Skipping coco annotation overlay in minimal mode"
elif [ ! -f "${IMAGE}" ]; then
    echo "Skipping coco annotation overlay since ${IMAGE} doesn't exist"
fi

echo -e "\n==================================================="
echo "Ground Truth saved to: ${OUTPUT_DIR}/${file_name}"
echo "==================================================="

# Run the inference commands
python3 main.py --root_dir ~/rtmp/data/shola/data/ \
    --tile_output_dir ./data/tiles \
    --val_dir ./data/val \
    --train_dir ./data/train \
    --test_dir ./data/test \
    --pipeline_config "${PIPELINE_CONFIG}" \
    --tile_size 2048 \
    --checkpoint_output_dir ./checkpoints/all \
    --training_image detectron2:1.3 \
    --inference_weights_path "${WEIGHTS}" \
    --inference_test_image "${IMAGE}" \
    --inference_output_dir "${OUTPUT_DIR}" \
    --inference_image detectron2-inference:1.4 \
    --inference_confidence_threshold ${CONFIDENCE} \
    ${GRADIO_MODE:+--inference_gradio_mode} \
    ${MINIMAL_VIS:+--inference_minimal_visualization} \
    ${CLASSES:+--inference_classes "${CLASSES}"} \
    --log_level INFO
