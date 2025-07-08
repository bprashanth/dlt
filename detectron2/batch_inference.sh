#!/bin/bash
# Batch inference script for the ForestFomo pipeline.
#
# This script runs inference on all images in the input directories and
# stitches the tiles. It manages its own pipeline config, so modifying the
# global pipeline config is useless. 
#
# Usage: ./batch_inference.sh -input_dirs "path1,path2,..." -output_dir <path> 
#     [-weights <path>] [-classes <class>] [-min]

# Default values
weights="./checkpoints/all/model_final.pth"
classes="Lantana Cover"
min=""
output_dir="./inference"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -input_dir) input_dirs="$2"; shift ;;
        -output_dir) output_dir="$2"; shift ;;
        -weights) weights="$2"; shift ;;
        -classes) classes="$2"; shift ;;
        -min) min="-min" ;;  # just a flag, doesn't take a value
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check required args
if [[ -z "$input_dirs" || -z "$output_dir" ]]; then
    echo "Usage: $0 -input_dir <path1,path2,...> -output_dir <path> [-weights <path>] [-classes <class>] [-min]"
    echo "Example: $0 -input_dir 'input/dir/1,input/dir/2' -output_dir ./inference"
    exit 1
fi

# Parse comma-separated input directories into an array
IFS=',' read -ra input_dir_array <<< "$input_dirs"

# Validate that all input directories exist
for dir in "${input_dir_array[@]}"; do
    if [[ ! -d "$dir" ]]; then
        echo "Error: Input directory '$dir' does not exist"
        exit 1
    fi
done

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
    "skip_stitching": true,
    "skip_offloading": true,
    "skip_metrics": true
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
    "skip_stitching": false,
    "skip_offloading": true,
    "skip_metrics": true
}
EOF
}

# Count total number of images across all input directories
total_images=0
for input_dir in "${input_dir_array[@]}"; do
    for image_file in "$input_dir"/*.{png,jpg,jpeg,JPG,JPEG,PNG}; do
        [ -e "$image_file" ] && ((total_images++))
    done
done

echo "Found $total_images images to process across ${#input_dir_array[@]} input directory(ies)"
echo "Input directories: ${input_dir_array[*]}"
echo "----------------------------------------"

# Step 1: Run inference on all images from all directories
echo "Step 1: Running inference on all images..."
create_inference_config

current=0
for input_dir in "${input_dir_array[@]}"; do
    echo "Processing images from directory: $input_dir"
    for image_file in "$input_dir"/*.{png,jpg,jpeg,JPG,JPEG,PNG}; do
        [ -e "$image_file" ] || continue
        ((current++))
        echo "Processing image $current/$total_images: $image_file"
        ./single_inference.sh -image "$image_file" -output_dir "$output_dir" -weights "$weights" -classes "$classes" -min -pipeline_config "$pipeline_config"
    done
done

# Step 2: Run stitching once for all images
echo "----------------------------------------"
echo "Step 2: Running stitching for all images..."
create_stitching_config
./single_inference.sh -output_dir "$output_dir" -weights "$weights" -pipeline_config "$pipeline_config"

# Clean up temporary directory
rm -rf "$tmp_dir"

echo "----------------------------------------"
echo "Completed processing all $total_images images from ${#input_dir_array[@]} directory(ies)"
