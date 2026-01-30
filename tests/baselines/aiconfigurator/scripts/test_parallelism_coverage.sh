#!/bin/bash
# Test Script: Parallelism Coverage
# Usage: ./test_parallelism_coverage.sh

set -e

CONFIG_FILE="tests/baselines/aiconfigurator/configs/coverage_exp.yaml"
OUTPUT_DIR="tests/baselines/aiconfigurator/results/coverage_run"

mkdir -p "$OUTPUT_DIR"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

echo "Running Parallelism Coverage Experiments..."
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"

aiconfigurator cli exp \
    --yaml_path "$CONFIG_FILE" \
    --save_dir "$OUTPUT_DIR"

echo "---------------------------------------------------"
echo "Coverage run completed."
ls -F "$OUTPUT_DIR"
