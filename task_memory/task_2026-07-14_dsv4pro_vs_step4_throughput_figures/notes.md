## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-14 | Initial notes from grilling session |
| 2026-07-15 | Clarified the production-code boundary and recorded v2 artifact provenance |
| 2026-07-15 | Added the no-source-edit constraint for the in-flight v3 refresh |
| 2026-07-15 | Recorded current-source v3 completion and authoritative archive paths |

# Notes: DSV4-Pro vs Step4 Throughput Figures

## v3 Execution Constraint and Completion

- Both throughput runner sources remained unchanged throughout all four v3 shards.
- Do not signal, restart, or edit the checkpoint of the independent GB300 full-matrix v3 process.
- Documentation and new result directories are allowed because they do not enter the runner-source fingerprint.
- Use `/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python` with `PYTHONPATH=src:.` and `MPLBACKEND=Agg`.

## Key Constraints

1. Step4 uses temporary MLA substitute for all 92 attention layers. ISL >= 65536 is approximation-dominated. Per D10, no visual marking required.

2. H800 (h800_sxm) is simulated SOL - no actual silicon measurements. All H800 data is model-predicted only.

3. The 21-config parallel space excludes EP=64 because Step4 has 352 experts, and `352/64 = 5.5` (uneven distribution). This is a known limitation documented in the parent task.

4. The ratio direction (step4/ds-v4-pro) means:
   - Ratio > 1: Step4 has higher throughput (Step4 is better)
   - Ratio < 1: DeepSeek-V4-Pro has higher throughput (DS is better)
   - This matches the reference figure visual: lines above baseline = baseline wins

5. Both figures use the SAME best config selected by output_token_throughput. The prefill throughput shown in fig1 is from this same config (not independently optimized for prefill).

## Relationship to Parent Task

This task builds on the completed `task_2026-07-10_step4_predefined_ops_plan` which:
- Implemented Step4 model in AIC (step4.py, config, family registration)
- Validated the 21-config parallel space
- Ran a 480-run matrix with different parameters (ISL/OSL/modes)
- All code changes are on the `step-design` branch

The current task does not modify production SDK behavior. It does:
- Add dedicated performance workflow and plotting source under `tests/performance/`
- Add unit coverage under `tests/unit/performance/`
- Runs new experiments with different ISL/OSL/mode parameters
- Produces visualization scripts and figures

The authoritative archive uses only `full-v3-*`, `figures_completed_v3`, and `figures_final_v3` artifacts. Earlier v1/v2 outputs are retained locally as historical evidence but are superseded for final staging. The v3 scientific payloads are exactly identical to v2 after normalizing only the strict execution-contract hash; v3 corrects provenance rather than numerical results.

## Reference Figure Style

From the provided image (prefill cost ratio):
- Matplotlib style with grid (dashed, light gray)
- One dashed baseline at y=1.0
- Multiple colored solid lines with circle markers
- Per-point text annotations above/below the marker
- Log-scale X axis
- Legend in upper-left
- Title at top
