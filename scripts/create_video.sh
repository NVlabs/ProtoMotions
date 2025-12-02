#!/bin/bash

# Video creation script for PNG sequence
INPUT_DIR="output/renderings/path_to_png_sequence"
OUTPUT_VIDEO="path_to_output_video.mp4"

echo "Creating video from PNG sequence..."
echo "Input directory: $INPUT_DIR"
echo "Output video: $OUTPUT_VIDEO"

# Create video using FFmpeg
# -framerate 30: 30 frames per second
# -i %04d.png: input pattern for 4-digit numbered PNG files
# -vf scale=3374:1370: scale to even dimensions for H.264 compatibility
# -c:v libx264: H.264 codec
# -preset medium: balance between encoding speed and compression
# -crf 18: high quality (lower = better quality, 18 is visually lossless)
# -pix_fmt yuv420p: pixel format for compatibility
ffmpeg -y -framerate 30 -i "$INPUT_DIR/%04d.png" \
       -c:v libx264 \
       -preset medium \
       -crf 25 \
       -pix_fmt yuv420p \
       "$OUTPUT_VIDEO"

if [ $? -eq 0 ]; then
    echo "Video created successfully: $OUTPUT_VIDEO"
    echo "Video details:"
    ffprobe -v quiet -print_format json -show_format -show_streams "$OUTPUT_VIDEO" | jq '.format.duration, .streams[0].width, .streams[0].height'
else
    echo "Error creating video"
    exit 1
fi 