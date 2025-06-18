#!/bin/bash

# Default values
weights="./checkpoints/all/model_final.pth"
classes="Lantana Cover"
min=""
output_dir="./inference"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -input_dir) input_dir="$2"; shift ;;
        -output_dir) output_dir="$2"; shift ;;
        -weights) weights="$2"; shift ;;
        -classes) classes="$2"; shift ;;
        -min) min="-min" ;;  # just a flag, doesn't take a value
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check required args
if [[ -z "$input_dir" || -z "$output_dir" ]]; then
    echo "Usage: $0 -input_dir <path> -output_dir <path> [-weights <path>] [-classes <class>] [-min]"
    exit 1
fi

# Create temporary directory for pipeline config
tmp_dir=$(mktemp -d)
pipeline_config="$tmp_dir/pipeline_config.json"

# Function to create pipeline config with inference enabled and stitching disabled
create_inference_config() {
    cat > "$pipeline_config" << EOF
{
    "skip_validation": true,
    "skip_tiling": true,
    "skip_annotation": true,
    "skip_training": true,
    "skip_inference": false,
    "skip_stitching": true
}
EOF
}

# Function to create pipeline config with inference disabled and stitching enabled
create_stitching_config() {
    cat > "$pipeline_config" << EOF
{
    "skip_validation": true,
    "skip_tiling": true,
    "skip_annotation": true,
    "skip_training": true,
    "skip_inference": true,
    "skip_stitching": false
}
EOF
}

# Count total number of images
total_images=0
for image_file in "$input_dir"/*.{png,jpg,jpeg,JPG,JPEG,PNG}; do
    [ -e "$image_file" ] && ((total_images++))
done

echo "Found $total_images images to process"
echo "----------------------------------------"

# Step 1: Run inference on all images
echo "Step 1: Running inference on all images..."
create_inference_config

current=0
for image_file in "$input_dir"/*.{png,jpg,jpeg,JPG,JPEG,PNG}; do
    [ -e "$image_file" ] || continue
    ((current++))
    echo "Processing image $current/$total_images: $image_file"
    ./single_inference.sh -image "$image_file" -output_dir "$output_dir" -weights "$weights" -classes "$classes" -min -pipeline_config "$pipeline_config"
done

# Step 2: Run stitching once for all images
echo "----------------------------------------"
echo "Step 2: Running stitching for all images..."
create_stitching_config
./single_inference.sh -output_dir "$output_dir" -weights "$weights" -pipeline_config "$pipeline_config"

# Clean up temporary directory
rm -rf "$tmp_dir"

echo "----------------------------------------"
echo "Completed processing all $total_images images"
