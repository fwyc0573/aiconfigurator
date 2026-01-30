# aiconfigurator Baseline Tests

This directory contains documentation, scripts, and configurations for using `aiconfigurator` as a research baseline.

## Structure

*   `docs/`: Detailed documentation and comparative analysis.
    *   `comparison_analysis.md`: Deep dive into simulation logic vs. Frontier.
    *   `usage_guide.md`: Practical guide for running tests.
    *   `test_gap_analysis.md`: Audit of test coverage and missing features.
    *   `model_config_loading.md`: Explanation of how model parameters are resolved.
    *   `web_ui_guide.md`: Instructions for launching and accessing the Web UI.
*   `scripts/`: Executable bash scripts for common scenarios.
    *   `test_default_cli.sh`: Runs the standard CLI default mode.
    *   `test_exp_mode.sh`: Runs the advanced experiment mode with YAML input.
    *   `test_parallelism_coverage.sh`: Runs extensive coverage tests (MoE, High-TP/PP).
*   `configs/`: Example configurations for experiment mode.
    *   `example_exp.yaml`: Basic aggregation/disaggregation test.
    *   `coverage_exp.yaml`: Advanced MoE and parallelism test cases.
*   `results/`: Directory for storing test outputs (git-ignored).

## Quick Start

1.  **Activate Environment**:
    ```bash
    source .venv_aiconfigurator/bin/activate
    ```

2.  **Run Default Test**:
    ```bash
    bash tests/baselines/aiconfigurator/scripts/test_default_cli.sh
    ```

3.  **Run Experiment Test**:
    ```bash
    bash tests/baselines/aiconfigurator/scripts/test_exp_mode.sh
    ```

4.  **Run Coverage Test**:
    ```bash
    bash tests/baselines/aiconfigurator/scripts/test_parallelism_coverage.sh
    ```

## Modification History

| Date       | Summary of Changes |
|------------|-------------------|
| 2026-01-09 | Added coverage tests and detailed documentation |
| 2026-01-09 | Initial setup of baseline testing structure |
