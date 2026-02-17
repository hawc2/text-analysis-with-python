#!/bin/bash
# Example script for running topic modeling on a restricted corpus
# Modify paths and parameters for your specific dataset

# Configuration
INPUT_DIR="Data/my_restricted_corpus"
OUTPUT_DIR="results/restricted_topic_modeling_$(date +%Y%m%d_%H%M%S)"
NUM_TOPICS=30
PASSES=25
WORKERS=4

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "================================================"
echo "Topic Modeling on Restricted Corpus"
echo "================================================"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Topics: $NUM_TOPICS"
echo "================================================"

# Run topic modeling
python scripts/batch_topic_modeling.py \
    --input-dir "$INPUT_DIR" \
    --file-pattern "*.txt" \
    --output-dir "$OUTPUT_DIR" \
    --num-topics "$NUM_TOPICS" \
    --passes "$PASSES" \
    --iterations 400 \
    --chunksize 200 \
    --workers "$WORKERS" \
    --save-corpus \
    --verbose

echo "================================================"
echo "Processing complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "================================================"
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "View visualization: open $OUTPUT_DIR/lda_visualization.html"
