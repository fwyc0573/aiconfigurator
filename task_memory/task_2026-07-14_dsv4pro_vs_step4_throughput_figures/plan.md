## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-14 | Initial plan drafted after grilling completion |
| 2026-07-14 | Aligned acceptance wording with strict `< 5000 ms` TTFT requirement |
| 2026-07-15 | Recorded completion of the corrected v2 execution and figure deliverables |
| 2026-07-15 | Added the current-source v3 refresh required after the runner contract changed |
| 2026-07-15 | Completed the current-source v3 refresh and all-GPU figure audit |

# Plan: DeepSeek-V4-Pro vs Step4 Throughput Ratio Figures

## Scope

Run a fresh disagg-only experiment matrix comparing Step4 and DeepSeek-V4-Pro across 4 GPU systems and 8 ISL values, then produce two ratio figures (prefill/decode throughput).

## Acceptance Criteria

1. All 64 matrix points (2 models × 4 systems × 8 ISL) complete or document infeasibility.
2. Best config for each (model, system, ISL) selected by highest output_token_throughput under strict TTFT < 5000ms.
3. fig1 (prefill) and fig2 (decode) rendered with correct ratio direction (step4/ds-v4-pro).
4. Every data point annotated with "gpu_type ratio_value".
5. X-axis log scale, step4=1.0 baseline dashed line, 4 colored GPU-type lines.
6. All experiments use uniform parameters: OSL=1024, disagg, 21-config space, SOL, all corrections=1.0.

## Phase 1: Experiment Script Development

- Create a dedicated runner script at `tests/performance/aic_roofline_pareto/run_dsv4pro_vs_step4_throughput.py`
- Adapt from existing `run_step4_comparison.py` framework
- Key modifications:
  - ISL set: [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
  - OSL: 1024 (fixed)
  - Serving mode: disagg only (skip agg)
  - TTFT SLA: single 5000ms target
  - Ranking metric: output_token_throughput
- Runner must output structured results (JSON/CSV) with per-config prefill and decode throughput

## Phase 2: Execution

- Run experiments for all 4 systems: gb300, h200_sxm, h100_sxm, h800_sxm
- System sharding strategy: can run systems in parallel (up to 3 concurrent)
- h800_sxm uses simulated SOL with `--systems-paths default,tests/performance/aic_roofline_pareto/systems`
- Store results in `task_memory/task_2026-07-14_dsv4pro_vs_step4_throughput_figures/results/`

## Phase 3: Best Config Selection & Ratio Computation

- For each (model, system, ISL): select rank-1 config by output_token_throughput descending
- Extract from best config:
  - prefill_input_throughput (for fig1)
  - output_token_throughput (for fig2)
- Compute ratios:
  - fig1_ratio[system][isl] = step4_prefill_throughput / dsv4pro_prefill_throughput
  - fig2_ratio[system][isl] = step4_output_throughput / dsv4pro_output_throughput

## Phase 4: Figure Generation

- Create plotting script at `tests/performance/aic_roofline_pareto/plot_throughput_ratio.py`
- Use matplotlib with style matching reference figure
- Generate two PNG/PDF figures:
  - `fig1_prefill_throughput_ratio.png`
  - `fig2_decode_throughput_ratio.png`
- Store in task_memory results directory

## Phase 5: Validation & Documentation

- Verify ratio computations manually for at least 2 sample points
- Document any infeasible/OOM points
- Update progress.md with execution evidence
- Write test report with numeric evidence

## Open Questions (None)

All design decisions resolved during grilling session. See requirements.md D1-D10.

## Completion Status

- Fresh corrected v2 shards: `64/64` mode-run identities complete.
- Interim menu: H200/H100/H800, `15` paired and `9` missing points.
- Final menu: GB300/H200/H100/H800, `23` paired and `9` missing points.
- Ratio audit: `46/46` stored prefill/decode ratios recomputed with maximum absolute delta `0.0`.
- Affected regression: `177/177` passed; Ruff and `git diff --check` passed.

## Current-Source v3 Refresh — Complete

The corrected-v2 artifacts remain historical evidence, but their checkpoint headers no longer match the current strict execution contract because the shared base runner changed after those runs. The current delivery therefore uses new immutable v3 directories and does not rebind, copy, or overwrite any v2 checkpoint.

1. [x] Keep the independent GB300 full-matrix v3 process running without signals, restarts, or checkpoint edits.
2. [x] Run current-source throughput v3 shards for H100, H800, and H200, then generate the three-system menu immediately.
3. [x] Run the current-source GB300 throughput v3 shard in the second compute lane without modifying the primary runner or checkpoint.
4. [x] Generate the four-system final menu from the four v3 shards.
5. [x] Validate exact contracts, `64/64` identities, `256/256` terminal outcomes, ratio recomputation, missing-point preservation, PNG metadata, and v2/v3 scientific identity.

Final current-source evidence: four shard contracts match current source and Git HEAD; mode-run identities=`64/64`; terminal outcomes=`256/256`; rank-one rows=`50`; paired/missing points=`23/9`; ratios=`46/46` with maximum recomputation delta=`0.0`; both PNGs=`2160 × 1260`, fully opaque, and byte-identical to v2. H800 remains simulated SOL.
