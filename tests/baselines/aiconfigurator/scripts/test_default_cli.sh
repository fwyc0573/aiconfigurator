#!/bin/bash
# Test Script: Default CLI Mode
# Usage: ./test_default_cli.sh [output_dir]

set -e

OUTPUT_DIR=${1:-"tests/baselines/aiconfigurator/results/default_run"}
mkdir -p "$OUTPUT_DIR"

echo "Running aiconfigurator in DEFAULT mode..."
echo "Model: QWEN3_32B, System: h200_sxm, GPUs: 32"

# Note: We use --database_mode SOL or EMPIRICAL if silicon data is missing in dev env.
# Assuming standard installation has data, trying SILICON (default) first.

aiconfigurator cli default \
    --model QWEN3_32B \
    --total_gpus 32 \
    --system h200_sxm \
    --ttft 300 \
    --tpot 10 \
    --isl 4000 \
    --osl 500 \
    --save_dir "$OUTPUT_DIR"

echo "---------------------------------------------------"
echo "Run completed. Results saved to $OUTPUT_DIR"
ls -F "$OUTPUT_DIR"
