#!/bin/bash

# Set directories
HOWTO_DIR="/store/swissai/a08/data/raw/howto100m/v2d_40k"
HDVILA_DIR="/store/swissai/a08/data/raw/hdvila/hd_vila_v2d"
OUTPUT_DIR="/store/swissai/a08/data/4m/video_rgb"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process HowTo100M files
for file in "$HOWTO_DIR"/*.tar; do
    filename=$(basename "$file")
    echo "Processing HowTo100M: $filename"
    
    # Add metadata
    echo "dataset: HowTo100M" > metadata.txt
    tar -rf "$file" metadata.txt
    rm metadata.txt
    
    # Copy to output directory
    cp "$file" "$OUTPUT_DIR/$filename"
done

# Get the last file number from HowTo100M
last_howto_num=$(ls "$HOWTO_DIR" | grep -oE '[0-9]+' | sort -n | tail -n 1)

# Process HD-VILA files
counter=$((last_howto_num + 1))
for file in "$HDVILA_DIR"/*.tar; do
    old_filename=$(basename "$file")
    new_filename=$(printf "%010d.tar" $counter)
    echo "Processing HD-VILA: $old_filename -> $new_filename"
    
    # Add metadata
    echo "dataset: HD-VILA" > metadata.txt
    tar -rf "$file" metadata.txt
    rm metadata.txt
    
    # Copy to output directory with new name
    cp "$file" "$OUTPUT_DIR/$new_filename"
    
    ((counter++))
done

echo "Merging complete. Output files are in $OUTPUT_DIR"