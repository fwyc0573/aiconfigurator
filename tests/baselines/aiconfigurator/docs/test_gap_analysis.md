# Test Gap Analysis

## Modification History

| Date       | Summary of Changes |
|------------|-------------------|
| 2026-01-09 | Initial gap analysis of aiconfigurator test suite |

## 1. Existing Coverage Audit

### 1.1 `test_default_cli.sh`
*   **Covered**: Basic CLI workflow, single dense model (`QWEN3_32B`), standard SLA constraints.
*   **Mode**: Default (Agg + Disagg comparison).
*   **Gap**: Uses default search space; doesn't stress-test edge cases or specific parallelism.

### 1.2 `test_exp_mode.sh`
*   **Covered**: Custom YAML configuration, explicit Agg/Disagg mode separation.
*   **Mode**: Experiment mode.
*   **Gap**: The example config was minimal (dense model only).

## 2. Identified Gaps

### 2.1 Missing Model Architectures
*   **MoE (Mixture of Experts)**: No tests for `DeepSeek-V3` or `Mixtral`. Critical for verifying Expert Parallelism (EP) logic and communication cost modeling (All-to-All).
*   **Large Context**: No tests with extremely long ISL (e.g., 128k) to stress-test KV cache memory modeling.

### 2.2 Missing Parallelism Strategies
*   **Pipeline Parallelism (PP)**: Default tests often stick to PP=1. Need to verify PP>1 logic and bubble overhead modeling.
*   **Expert Parallelism (EP)**: Specific to MoE, untested in baselines.
*   **High-Degree Parallelism**: Validating logic for >8 GPUs (multi-node scenarios).

### 2.3 Advanced Features
*   **MTP / Speculative Decoding**: `nextn` parameters are present in code but untested.
*   **WideEP**: `enable_wideep` flag in config is untested.
*   **Database Modes**: Only `SILICON` is implicitly tested. `SOL`, `EMPIRICAL`, `HYBRID` modes need explicit verification.

## 3. Enhancement Plan

### 3.1 New Coverage Script
Created `test_parallelism_coverage.sh` and `coverage_exp.yaml` to address:
*   **MoE**: Added `DeepSeek-V3` experiments.
*   **EP**: Configured `moe_ep_list: [1, 2, 8]`.
*   **PP/TP**: Added `QWEN3_32B` with `pp_list: [1, 2, 4]` and `tp_list: [8]`.

### 3.2 Future Recommendations
*   Add a specific test for **Database Modes** (`--database_mode SOL`) to verify fallback logic when silicon data is missing.
*   Add a test for **Constraint Boundaries** (e.g., impossible SLAs) to verify error handling/reporting.
