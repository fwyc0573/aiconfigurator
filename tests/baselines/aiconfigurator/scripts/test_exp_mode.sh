#!/bin/bash
# Test Script: Experiment Mode
# Usage: ./test_exp_mode.sh [config_file]

set -e

CONFIG_FILE=${1:-"tests/baselines/aiconfigurator/configs/example_exp.yaml"}
OUTPUT_DIR="tests/baselines/aiconfigurator/results/exp_run"

mkdir -p "$OUTPUT_DIR"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

echo "Running aiconfigurator in EXPERIMENT mode..."
echo "Config: $CONFIG_FILE"

aiconfigurator cli exp \
    --yaml_path "$CONFIG_FILE" \
    --save_dir "$OUTPUT_DIR"

echo "---------------------------------------------------"
echo "Run completed. Results saved to $OUTPUT_DIR"
ls -F "$OUTPUT_DIR"
