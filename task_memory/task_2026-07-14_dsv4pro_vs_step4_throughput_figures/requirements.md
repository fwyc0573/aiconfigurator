## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-14 | Initial requirements captured from grilling session |
| 2026-07-14 | Clarified TTFT SLA as strict `< 5000 ms` per user instruction |
| 2026-07-15 | Captured the temporary priority to publish non-GB300 figures first and finish all-system figures after GB300 |
| 2026-07-15 | Captured approval for the current-source rerun and minimum-cost branch integration |

# Requirements: DeepSeek-V4-Pro vs Step4 Throughput Ratio Figures

## Original Request

[Original Request] 严格参考已有的 "Prefill cost ratio vs baseline" 图的绘制方式，补充独立的 2 张图，比较 DeepSeek-V4-Pro 和 Step4 的吞吐量（不同颜色的线代表不同 GPU type），Step4 为 baseline。

## Temporary Priority Adjustment

[Original Request] 临时调整任务：1。让gb300继续后台运行，完成后进行所有gpu type的figs绘制 2。**new**：先绘制已经完成的system(除了gb300以外)的figs，并规范整理到一个menu下方便我查阅

## Rerun and Integration Approval

1. [Original Request] 批准重跑 current-source shards，以修复 strict execution-contract provenance mismatch。
2. [Original Request] 使用最小代价完成当前 worktree branch 的整理、提交、push 和 merge，减少不必要的额外运行和测试验证。

## Grilling Decisions (D1-D10)

### D1: Y-axis format
- **Decision**: Ratio format (strict reference to original figure style)
- Ratio definition: `step4 throughput / ds-v4-pro throughput`
- step4 = 1.0 baseline dashed line
- Each GPU type is one colored line

### D2: OSL (Output Sequence Length)
- **Decision**: Unified OSL=1024 for both figures
- Both figures derived from the same experiment (split prefill/decode from one run)
- Same best config applies to both fig1 and fig2

### D3: Best config selection criterion
- **Decision**: By E2E throughput (output_token_throughput, tokens/s/gpu_cluster)
- Under strict constraint TTFT < 5000ms
- Highest output_token_throughput config selected as "best" for each (model, system, ISL) combination

### D4: Parallel search space
- **Decision**: 21-config (Pattern A 17 + Pattern B 4), total_gpus=64
- Reuses the validated space from the step4-predefined-ops task
- No EP=64, max worker_gpus=32

### D5: E2E throughput metric
- **Decision**: output_token_throughput (tokens/s/gpu_cluster)
- Used as ranking metric for best config selection

### D6: Figure annotations
- **Decision**: GPU type name + ratio value on each data point
- Style follows reference figure annotation positioning
- No cost annotations

### D7: GPU system mapping
- **Decision**: Confirmed one-to-one mapping
  - Figure display: gb300, h200, h100, h800
  - AIC system name: gb300, h200_sxm, h100_sxm, h800_sxm
  - h800 remains simulated SOL (no silicon data)

### D8: Data freshness
- **Decision**: All experiments run fresh
- No reuse of existing ISL=4096 OSL=1024 decode_smoke data
- Ensures uniform experimental conditions across all ISL points

### D9: Ratio direction (polarity)
- **Decision**: ratio = step4 / ds-v4-pro
- ratio > 1 → step4 is faster (step4 dominates)
- ratio < 1 → ds-v4-pro is faster
- ratio = 1.0 → break even (dashed baseline)
- Visual pattern matches reference: lines above baseline = baseline (step4) is better

### D10: MLA approximation handling
- **Decision**: No special marking, display directly
- Known that Step4 uses temporary MLA substitute for all 92 attention layers
- ISL >= 65536 is approximation-dominated but not visually distinguished

## Fixed Experiment Parameters

| Dimension | Value |
|---|---|
| Models | Step4 (baseline), DeepSeek-V4-Pro |
| Backend | vllm |
| ISL (sequence lengths) | 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072 |
| OSL | 1024 |
| Serving mode | disagg only |
| TTFT SLA | < 5000 ms (strict) |
| TPOT | No constraint (50000 ms) |
| Parallel space | 21-config (Pattern A 17 + Pattern B 4) |
| Total GPUs | 64 |
| pp | 1 |
| cp | 1 |
| tp | <= 8 |
| Database mode | SOL |
| Systems | gb300, h200_sxm, h100_sxm, h800_sxm |
| prefix | 0 |
| nextn | 0 |
| chunked prefill | disabled (prefill_enable_chunked_prefill=false) |
| Correction factors | prefill_latency_correction=1.0, decode_latency_correction=1.0, rate_match_prefill_degradation=1.0, rate_match_decode_degradation=1.0, autoscale_ttft_correction_factor=1.0 |
| Batch sweep | From 1, step 1, cap-aware doubling |
| pareto_sweep | false |

## Expected Output

### fig1: Prefill Throughput Ratio
- Title: "Prefill throughput ratio vs baseline (step4)"
- X-axis: Sequence length (tokens), log scale: 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
- Y-axis: Relative throughput vs step4 (ratio = step4 prefill throughput / ds-v4-pro prefill throughput)
- Lines: 4 colored lines (gb300, h200, h100, h800), each showing ds-v4-pro relative to step4
- Baseline: step4 = 1.0 dashed line
- Annotations: "gpu_type ratio_value" on each data point
- Metric: prefill_input_throughput of the best config (selected by output_token_throughput ranking)

### fig2: Decode Throughput Ratio
- Title: "Decode throughput ratio vs baseline (step4)"
- X-axis: Sequence length (tokens), log scale: 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
- Y-axis: Relative throughput vs step4 (ratio = step4 decode throughput / ds-v4-pro decode throughput)
- Lines: 4 colored lines (gb300, h200, h100, h800), each showing ds-v4-pro relative to step4
- Baseline: step4 = 1.0 dashed line
- Annotations: "gpu_type ratio_value" on each data point
- Metric: output_token_throughput of the best config (selected by output_token_throughput ranking)

## Experiment Scale

- 2 models × 4 systems × 8 ISL values = 64 matrix points
- Each point runs disagg mode only, with the full 21-config parallel space
- Disagg uses AA/AB/BA/BB Pattern pairings (441 base pairs per point before batch enumeration)
- Total mode-run estimate: 64 disagg tasks × 4 pattern combos = 256 mode runs (before cap-doubling reruns)

## Derivation from Existing Infrastructure

This task reuses the validated infrastructure from `task_2026-07-10_step4_predefined_ops_plan`:
- Step4 model class, config, and ops pipeline
- DeepSeek-V4-Pro model and ops pipeline
- The 21-config parallel space materialization
- The disagg search and evaluation framework
- The cap-aware batch sweep logic
- The result normalization and ranking pipeline

Key differences from the existing 480-run matrix:
1. ISL set: 8 points (1024-131072) instead of 5 (4096-1048576)
2. OSL: 1024 instead of 1 (primary) or 1024 (decode_smoke only at ISL=4096)
3. Serving mode: disagg only (not both agg and disagg)
4. TTFT SLA: single 5000ms threshold (not 5 targets: 200/500/1000/2000/5000)
5. Output: throughput ratio figures instead of full comparison tables
