## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-15 | Recorded final independent corrected-v3 APPROVE and released exact staging/integration |
| 2026-07-15 | Completed formatter-consistent corrected-v3 merge and advanced Phase 6 to minimum-cost verification and delivery |
| 2026-07-15 | Advanced Phase 6 through corrected-v2 GB300 completion, four-system merge, strict resume, and semantic verification |
| 2026-07-14 | Completed final static and completion-verification reruns after correcting ANSI log parsing |
| 2026-07-14 | Completed the final independent completion review and closed all delivery gates |
| 2026-07-10 | Created draft implementation and validation plan for Step4 predefined ops |
| 2026-07-10 | Updated attention plan after grilling accepted `sliding_window_512` |
| 2026-07-12 | Corrected attention topology and FFN plan using Step4Air YAML evidence |
| 2026-07-12 | Finalized gated SwiGLU dense FFN after user adjudication |
| 2026-07-12 | Finalized shared-expert precision and merge behavior after user adjudication |
| 2026-07-13 | Finalized checkpoint nextn=3 with normalized nextn=0 comparison policy |
| 2026-07-13 | Added user-directed temporary MLA attention approximation pending scope clarification |
| 2026-07-13 | Fixed temporary MLA coverage at all 92 layers with retained audit labels |
| 2026-07-13 | Added independent WATCH conditions for the candidate MLA geometry |
| 2026-07-13 | Fixed temporary MLA geometry at the DeepSeek-V3 generic baseline |
| 2026-07-13 | Fixed local Step4 model identity and dedicated family naming |
| 2026-07-13 | Fixed H800 validation to the test-side simulated SOL system spec |
| 2026-07-13 | Fixed the requested sequence matrix as binary ISL values |
| 2026-07-13 | Fixed primary `osl=1` matrix and independent `4K/1024` decode smoke |
| 2026-07-13 | Confirmed `step-design` as the implementation branch |
| 2026-07-13 | Fixed all required validation workloads to explicit `prefix=0` |
| 2026-07-13 | Fixed required serving-mode coverage to both `agg` and `disagg` |
| 2026-07-13 | Fixed TTFT-only SLA sweep and `tokens/s/gpu_cluster` ranking from the Phase 2 reference |
| 2026-07-13 | Disabled chunked prefill explicitly across both required serving modes |
| 2026-07-13 | Added uneven-EP feasibility gate for Step4 `352 experts / EP=64` |
| 2026-07-13 | Finalized the common 21-config parallel space and exhaustive batch-cap policy |
| 2026-07-13 | Replaced remaining path/smoke placeholders with exact later-implementation targets |
| 2026-07-13 | Repaired search-space invariants, role-specific realization, worker orchestration, topology, OOM, and roofline-correction policy after reference parity audit |
| 2026-07-13 | Activated implementation after user authorization and completed the clean-baseline gate |
| 2026-07-13 | Fixed deterministic ranking and Step4-minus-DeepSeek delta semantics after independent review |
| 2026-07-13 | Revalidated reference-search parity and aligned primary `OSL=1` ranking with the accepted Prefill-throughput decision |
| 2026-07-14 | Reopened execution for the user-approved ISSUE-048 aggregate semantic remediation and review gates |
| 2026-07-14 | Added the user-approved post-review ISSUE-049 through ISSUE-052 strict-remediation gate |
| 2026-07-14 | Expanded the matrix gate with the independently audited search-evaluation, SOL admission, collective provenance, TPOT, and engine-backend contracts |
| 2026-07-14 | Marked ISSUE-048 through ISSUE-052 focused remediation complete while retaining broader, artifact, review, full-unit, and matrix gates |
| 2026-07-14 | Recorded Gate 6/7 strict-TDD completion and Gate 8/9/10 read-only audit status before the remaining architecture gate |
| 2026-07-14 | Accepted the independent Gate 8 architecture APPROVE verdict and opened strict RED implementation without renewed scope approval |
| 2026-07-14 | Restored three execution-level search-space clarifications after cross-worktree parity review |
| 2026-07-14 | Reconciled conflicting Gate 8 reviews and bound the final `CONTEXT_CAPTURE` architecture |
| 2026-07-14 | Completed Gate 8-10 remediation and focused/static verification before fresh representative artifacts |
| 2026-07-14 | Bound the ISSUE-059 zero-baseline TPOT contract after real H200 failure and independent APPROVE review |
| 2026-07-14 | Completed ISSUE-059 strict RED/GREEN, combined regression, and static verification |
| 2026-07-14 | Recorded Gate 12 independent post-change APPROVE and retained the stricter full-unit-before-matrix ordering |
| 2026-07-14 | Completed fresh H200 primary/decode, strict resume, and real `2+2` shard representative gates |
| 2026-07-14 | Completed Gate 13 post-change full-unit regression and unblocked Gate 14 matrix execution |
| 2026-07-14 | Bound Gate 14 system shards to one independently approved full-matrix execution contract |
| 2026-07-14 | Started Gate 14 Wave 1 system shards at the approved maximum concurrency of three |
| 2026-07-14 | Completed the H100 Wave 1 shard and strict-resume proof while GB300/H200 continue |
| 2026-07-14 | Completed the H200 Wave 1 shard and strict-resume proof while GB300 continues |
| 2026-07-14 | Completed all three Wave 1 shards and strict-resume proofs; released simulated H800 Wave 2 |
| 2026-07-14 | Started the single-lane simulated-SOL H800 Wave 2 shard |
| 2026-07-14 | Completed simulated-SOL H800 and all four shard strict-resume gates; released full merge |
| 2026-07-14 | Completed the strict 480-run merge, merged resume proof, semantic validation, and numeric comparison summary |
| 2026-07-14 | Completed the corrected final Ruff, format, diff, and task-doc whitespace gate |
| 2026-07-15 | Reopened the completed task for branch review, archival commit, remote push, and `step-design` integration |

# Plan: Step4 Predefined Ops in AIC

## Scope

This plan is the active implementation contract for adding Step4 predefined ops to AIC. The user authorized execution after the docs/grilling gate completed. Implementation ran in the isolated `task/step4-predefined-ops` worktree while the original dirty `step-design` checkout remained untouched. On 2026-07-15, the user explicitly extended the task to review, archive, commit, and push the feature branch, then safely reconcile and merge it into the sibling `step-design` worktree.

## Acceptance Criteria

1. Step4 can be resolved from a predefined local model config without network access.
2. Step4 has an explicit model family and model ops pipeline consistent with existing AIC conventions.
3. Step4 op latency estimation uses roofline-only behavior in the required validation flow.
4. Step4 does not depend on DS-V4 profiling-based perfdb or DS-V4 measured MegaMoE module data.
5. Step4 supports the requested matrix on `vllm`, `total_gpus=64`, `pp=1`, `tp<=8`, with `ep>8` and GPU workers `>8` allowed by the search/config path.
6. Step4 and DS-V4-Pro outputs are compared over the same validation matrix with numeric evidence.
7. Unit tests cover parsing, model registration, op counts, roofline-only behavior, and failure modes.
8. Integration/e2e tests cover CLI or SDK path generation and representative matrix runs.
9. Every result produced with the temporary MLA substitute is explicitly labeled as an approximation and is not presented as faithful Full/SWA Step4 attention.

## Accepted Design Direction

### Model family

Create a dedicated Step model family rather than forcing Step4 into `MOE` or `DEEPSEEKV4`.

Accepted local naming contract:

- `ModelFamily`: `STEP4`
- Architecture mapping: `Step4ForCausalLM -> STEP4`
- Model class file: `src/aiconfigurator/sdk/models/step4.py`
- Config file: `src/aiconfigurator/model_configs/stepfun-ai--Step4_config.json`
- Model ID: `stepfun-ai/Step4`

This is an AIC local naming contract. Do not claim it is a verified public HuggingFace checkpoint identity unless a canonical source is supplied later.

Rationale:

- Step4 combines full + non-full attention counts, first-N dense FFN layers, shared expert, routed MoE, and no DS-V4 mHC.
- It is closer to Step3.7 / Step3p5 than to DS-V4-Pro.
- A dedicated class avoids polluting generic `MOEModel` with Step-specific dense-layer and layer-type scheduling rules.

### Attention ops

Target architecture corrected after reviewing both Step4Air YAML files:

- `23` full-attention layers:
  - GQA with `64` query heads and `8` KV heads;
  - `qk_head_dim=128`, `v_head_dim=128`;
  - unlimited/full context (`window_size=0` or the AIC full-attention equivalent).
- `69` non-full layers:
  - GQA SWA with `96` query heads and `8` KV heads;
  - `qk_head_dim=128`, `v_head_dim=128`;
  - `window_size=512`.
- Both variants use FP8 Q/K/V/output projection intent and FP8 KV-cache read/append intent.
- Prefill sequence sharding from the YAML is a runtime parallel choice, not a model constant.

The two variants must not share one fixed head topology with only `window_size` changed.

Latest user direction for the temporary implementation model:

- Use MLA as the temporary attention substitute and declare that approximation explicitly.
- Do not reinterpret MLA as the actual Step4 architecture; retain the Full/SWA YAML topology above as the unresolved target.
- Use only a direct `DatabaseMode.SOL` MLA path. Do not query profiling-based MLA perfdb data.
- Apply the MLA SOL substitute to all `92` layers. Preserve two separately named/countable groups (`23 Full`, `69 SWA`) for architecture traceability, while documenting that both currently share one latency approximation and therefore cannot establish a Full-vs-SWA performance difference.
- Use the accepted DeepSeek-V3 generic MLA geometry: heads=`128`, q_lora=`1536`, kv_lora=`512`, qk_nope=`128`, qk_rope=`64`, v=`128`; retain Step4 H=`4096`.
- Model each temporary layer with explicit SOL projection dimensions `4096->2112`, `1536->24576`, `512->32768`, and `16384->4096`, plus granular MLA attention/BMM core operations. Every dimension must be asserted structurally in unit tests.
- Validation must separately assert `PerformanceResult.source == "sol"` for the temporary MLA attention path and include an approximation marker in every Step4-vs-DS-V4-Pro result artifact.
- If the DeepSeek-V3-style geometry is accepted, model explicit projection GEMMs plus the granular MLA core; do not use `FallbackOp` or the module-level profiled abstraction.
- Matrix artifacts must include separate attention and non-attention latency contributions. Rows at `isl>=64k` must carry an `approximation_dominated` marker because the substituted full-context sequence term is at least `128x` the real SWA window term before roofline/bottleneck effects.
- Preserve separately named Full-MLA-approx and SWA-MLA-approx op groups so the future window-aware replacement can target the `69` SWA layers without changing the `23` Full labels.
- Treat `ISL/512` only as an analytical attention-term ratio. Do not present it as predicted total error or measured vLLM discrepancy.

Avoid in first implementation:

- DS-V4-specific measured attention modules or any model path that silently selects profiling data.
- MSA/DSA/custom sparse ops that may rely on empirical/profiling transfer.
- Any profiling-based `perfdb` path for Step4-specific latency estimates.

### FFN / MoE ops

Resolved structure and precision behavior:

- First `4` trunk layers are a contiguous dense prefix using `dense_inter_size=13824`.
- Dense FFN uses gated SwiGLU: combined gate/up projection, `SiLU(gate) * up`, then down projection.
- The decision follows the CSV's exact `H * I * 3` parameter formula rather than AELLO's omission-derived ungated default.
- Per-layer projection parameters are `4096 * 13824 * 3 = 169,869,312`; four layers total `679,477,248`.
- Dense norm is BF16; dense up/down projections are intended FP8.
- Remaining 88 trunk layers: routed MoE with:
  - `num_experts=352`
  - `topk=8`
  - `moe_inter_size=1536`
  - `shared_expert_inter_size=1536`
- Routed expert algorithm is gated SiLU: combined gate/up projection, `SiLU(gate) * up`, then down projection.
- Routed dataflow includes BF16 norm/router/combine, FP8 dispatch/expert projections, and explicit combine.
- Include the CSV-mandated shared expert as a separate gated branch with FP8 gate/up/down projections and BF16 norm.
- Merge the routed and shared outputs in BF16. This is an explicit user decision because the YAML examples omit the shared branch.

### MTP / nextn

Accepted policy:

1. Declare `num_nextn_predict_layers=3` in the Step4 predefined checkpoint config.
2. Set `nextn=0` explicitly for both Step4 and DS-V4-Pro throughout the primary comparison matrix so the result isolates trunk ops.
3. Add a separate Step4 `nextn=3` structural/smoke check to verify checkpoint-default resolution and generation-op construction.
4. Do not present the separate MTP check as a Step4-vs-DS-V4-Pro TPOT comparison.

Do not infer AIC's full `nextn_accept_rates` list from AELLO's single `acceptance_rate=0.80`; that mapping is a separate runtime-modeling decision.

## Planned File Changes for Later Implementation

No files below are changed in the current docs-only stage.

### Likely source changes

- `src/aiconfigurator/sdk/common.py`
  - Add `STEP4` model family, `Step4ForCausalLM` architecture mapping, `Step4Config`, and `stepfun-ai/Step4` in `DefaultHFModels`.
- `src/aiconfigurator/sdk/utils.py`
  - Parse Step4/StepMoE config fields into `extra_params`.
  - Validate layer schedule lengths and fail fast for inconsistent configs.
- `src/aiconfigurator/sdk/models/step4.py`
  - Build context and generation ops with split counts for full/sliding attention and dense/MoE FFN.
  - Do not use DS-V4-specific measured modules.
- `src/aiconfigurator/model_configs/stepfun-ai--Step4_config.json`
  - Local predefined config built from CSV and resolved grilling decisions.
- `src/aiconfigurator/sdk/models/__init__.py` does not need a functional registry edit because package auto-discovery imports `step4.py`; add a `Step4Model` re-export only if the implementation test requires package-root import compatibility.

### Likely test changes

- `tests/unit/sdk/test_common.py`
  - Verify `STEP4`, `Step4ForCausalLM`, and `stepfun-ai/Step4` registration/default-model metadata.
- `tests/unit/sdk/test_utils.py`
  - Verify local Step4 config parsing, exact architecture fields, and fail-fast behavior for inconsistent layer schedules.
- `tests/unit/sdk/models/test_step4.py`
  - New focused unit suite for Step4 op structure, precision boundaries, cross-composition, SOL source, and failure modes.
  - Verify Step4 config declares `num_nextn_predict_layers=3` and resolves checkpoint-default `nextn=3` when no runtime override is supplied.
  - Verify `23 full GQA(64/8) + 69 sliding GQA(96/8, window_size=512)`.
  - Verify `4 dense + 88 MoE`, routed/shared intermediate size `1536`, and one shared branch per MoE layer.
  - Verify qk/v head dimensions are both `128` for both attention variants.
  - Verify effective attention projection/output and KV-cache read/append precision is FP8.
  - Verify dense norm BF16, gate/up/down FP8, and gated SwiGLU graph.
  - Verify routed MoE norm/router/combine BF16 and dispatch/expert projections FP8.
  - Verify shared-expert norm BF16, gate/up/down FP8, gated SwiGLU graph, and BF16 routed/shared merge.
  - Verify exact cross-composition after the dense-prefix/SWA rule: `4 dense+SWA`, `23 MoE+full`, `65 MoE+SWA`, total `92`.
  - Reconcile full/SWA Q/K/V/O parameter counts computed from their distinct head topology with the CSV per-attention summary; report expected, computed, absolute delta, and relative delta without calibration factors.
  - Reconcile the approved dense FFN graph against CSV `param per dense=169,869,312` and record the same numeric evidence.
- Unit tests for roofline-only behavior:
  - Force `DatabaseMode.SOL` and assert Step4 does not instantiate DS-V4 measured modules.
- `tests/unit/sdk/models/test_mtp_scaling.py`
  - Verify the primary Step4 and DS-V4-Pro matrix tasks both carry explicit `nextn=0`.
  - Verify a separate Step4 `nextn=3` structural/smoke point builds generation ops without making a cross-model performance assertion.
- `tests/performance/aic_roofline_pareto/run_step4_comparison.py`
  - New reproducible SOL-only Step4/DeepSeek-V4-Pro matrix runner using the accepted workloads, common `21` configs, repeated cap expansion, and result/report schema.
- `tests/unit/performance/test_step4_roofline_matrix.py`
  - Unit-test exact `17+4=21` pattern materialization, aggregate/disaggregate search construction, `AA/AB/BA/BB` pairings, cap-expansion branches, terminal conditions, ranking exclusion for saturated rows, and `240/480` matrix arithmetic.
- `tests/e2e/cli/test_cli_step4_generate.py`
  - E2E smoke the local model-config resolution and `vllm` SOL generation path without network access.
- `task_memory/task_2026-07-10_step4_predefined_ops_plan/test_report_YYYY-MM-DD_step4_predefined_ops.md`
  - Record exact environment, commands, PASS/FAIL evidence, matrix counts, representative latency/throughput values, comparison deltas, cap behavior, and known approximation labels during the later implementation pass.

## Test Matrix Plan for Later Implementation

### Matrix dimensions

| Dimension | Values |
|---|---|
| model | Step4, DeepSeek-V4-Pro |
| hardware/system | `gb300`, `h200_sxm`, `h100_sxm`, `h800_sxm` |
| backend | `vllm` |
| serving mode | `agg`, `disagg` |
| input sequence length | `4096`, `16384`, `65536`, `262144`, `1048576` |
| output sequence length | explicit `1` for all five primary-matrix points |
| prefix-cache length | explicit `0` for every required workload |
| TTFT SLA | `200`, `500`, `1000`, `2000`, `5000` ms |
| TPOT constraint | none; encode explicitly as `tpot=50000` ms with `pareto_sweep=false` |
| ranking objective | primary `OSL=1`: descending cluster-normalized input/prefill token throughput; `4K/1024` decode smoke: descending output `tokens/s/gpu_cluster` |
| chunked prefill | explicit `false` for both aggregate and disaggregate prefill paths |
| total GPUs | `64` |
| pp | `1` |
| tp | `1`, `2`, `4`, `8` where valid |
| ep | Pattern B=`1`; Pattern A=`2`, `4`, `8`, `16`, `32`; no `64` |
| GPU workers | `1`, `2`, `4`, `8`, `16`, `32` as materialized by the accepted Pattern A/B spaces |
| database mode | `SOL` |
| empirical correction/scaling | explicitly set all five latency/rate-match/autoscale factors to `1.0` |
| nextn | explicit `0` for both models in the primary comparison matrix |

Separate from the primary matrix, run one Step4-only checkpoint-default `nextn=3` smoke at `h200_sxm`, `vllm`, aggregate mode, `total_gpus=64`, `isl=4096`, `osl=1024`, `prefix=0`, TTFT=`5000` ms, `tpot=50000` ms, `pareto_sweep=false`, chunked prefill disabled, and `DatabaseMode.SOL`. Pin the parallel row to accepted Pattern B with `tp=8`, `moe_tp=8`, `moe_ep=1`, `dp=1`, `pp=1`, and `cp=1`. This is structural/smoke evidence only and must not be ranked against DeepSeek-V4-Pro.

Also run a distinct Step4-vs-DS-V4-Pro decode smoke at `isl=4096, osl=1024`, with explicit `nextn=0` and `prefix=0` for both models. This point covers sustained generation ops and is not part of the five-point long-ISL primary matrix or the Step4-only `nextn=3` structural check. It participates in the same complete `4 systems x 5 TTFT targets x 2 serving modes` expansion as each primary workload.

Expected comparison size before separate Step4 `nextn=3` smoke coverage:

```text
2 models × 4 systems × 6 workloads × 5 TTFT targets = 240 matrix points
240 matrix points × 2 serving modes = 480 mode-run rows
```

### Accepted parallel-search feasibility policy

Use the same exact `21` AIC-materialized configurations for Step4 and DeepSeek-V4-Pro in every aggregate, disaggregate-prefill, and disaggregate-decode search.

#### Enumeration invariants

The search contract is defined by the materialized rows, not only by broad candidate lists:

```text
worker_gpus = tp * dp * pp * cp = tp * dp
dp * tp * cp = moe_tp * moe_ep
pp = 1
cp = 1
tp <= 8
```

- Pattern A, pure EP: `moe_tp=1`, `moe_ep=tp*dp`, and `tp*dp>=2`.
- Pattern B, pure MoE-TP: `dp=1`, `moe_tp=tp`, and `moe_ep=1`.
- vLLM forbids simultaneous `moe_tp>1` and `moe_ep>1`; no accepted row may violate that condition.
- `worker_gpus` is the GPU width of one worker. It is not the same as the number of prefill/decode workers or deployment replicas.
- The Pattern A/B split enforces these equalities by construction; do not generate a broader Cartesian product and post-filter it afterward.

Pattern A is pure EP and materializes `17` valid configurations after AIC filtering:

```yaml
num_gpu_candidates: [2, 4, 8, 16, 32]
tp_candidates: [1, 2, 4, 8]
pp_candidates: [1]
dp_candidates: [1, 2, 4, 8, 16, 32]
moe_tp_candidates: [1]
moe_ep_candidates: [2, 4, 8, 16, 32]
cp_candidates: [1]
```

Pattern B is pure MoE-TP and materializes `4` configurations:

```yaml
num_gpu_candidates: [1, 2, 4, 8]
tp_candidates: [1, 2, 4, 8]
pp_candidates: [1]
dp_candidates: [1]
moe_tp_candidates: [1, 2, 4, 8]
moe_ep_candidates: [1]
cp_candidates: [1]
```

The exact aligned total is `17 pure-EP + 4 pure-MoE-TP = 21` configurations. The space fixes `pp=1`, `cp=1`, `tp<=8`, reaches EP=`32` and worker size=`32`, and therefore exercises both EP and worker sizes above `8` without introducing inaccurate Step4 uneven-EP rows. Disaggregate mode retains the `AA`, `AB`, `BA`, and `BB` Pattern A/B pairings between prefill and decode.

#### Authoritative 21-row materialization

| # | Pattern | tp | dp | pp | moe_tp | moe_ep | cp | worker_gpus |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | B | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2 | B | 2 | 1 | 1 | 2 | 1 | 1 | 2 |
| 3 | B | 4 | 1 | 1 | 4 | 1 | 1 | 4 |
| 4 | B | 8 | 1 | 1 | 8 | 1 | 1 | 8 |
| 5 | A | 1 | 2 | 1 | 1 | 2 | 1 | 2 |
| 6 | A | 1 | 4 | 1 | 1 | 4 | 1 | 4 |
| 7 | A | 1 | 8 | 1 | 1 | 8 | 1 | 8 |
| 8 | A | 1 | 16 | 1 | 1 | 16 | 1 | 16 |
| 9 | A | 1 | 32 | 1 | 1 | 32 | 1 | 32 |
| 10 | A | 2 | 1 | 1 | 1 | 2 | 1 | 2 |
| 11 | A | 2 | 2 | 1 | 1 | 4 | 1 | 4 |
| 12 | A | 2 | 4 | 1 | 1 | 8 | 1 | 8 |
| 13 | A | 2 | 8 | 1 | 1 | 16 | 1 | 16 |
| 14 | A | 2 | 16 | 1 | 1 | 32 | 1 | 32 |
| 15 | A | 4 | 1 | 1 | 1 | 4 | 1 | 4 |
| 16 | A | 4 | 2 | 1 | 1 | 8 | 1 | 8 |
| 17 | A | 4 | 4 | 1 | 1 | 16 | 1 | 16 |
| 18 | A | 4 | 8 | 1 | 1 | 32 | 1 | 32 |
| 19 | A | 8 | 1 | 1 | 1 | 8 | 1 | 8 |
| 20 | A | 8 | 2 | 1 | 1 | 16 | 1 | 16 |
| 21 | A | 8 | 4 | 1 | 1 | 32 | 1 | 32 |

Materialization acceptance numbers are fixed at: Pattern A=`17`, Pattern B=`4`, union=`21`, `EP>8` rows=`8`, `EP=16` rows=`4`, `EP=32` rows=`4`, and maximum `worker_gpus=32`.

#### Role-prefixed Task realization

Pattern A and Pattern B must be separate experiments. Combining both into one broad candidate bundle would allow `enumerate_parallel_config()` to create extra rows such as `dp>1, moe_ep=1, moe_tp=tp*dp`; those rows do not represent the intended vLLM worker model. Do not create invalid rows and then remove them by post-processing.

Aggregate mode uses two Task YAMLs with these exact role-prefixed fields:

```yaml
agg_patternA:
  agg_num_gpu_candidates: [2, 4, 8, 16, 32]
  agg_tp_candidates: [1, 2, 4, 8]
  agg_pp_candidates: [1]
  agg_dp_candidates: [1, 2, 4, 8, 16, 32]
  agg_moe_tp_candidates: [1]
  agg_moe_ep_candidates: [2, 4, 8, 16, 32]
  agg_cp_candidates: [1]

agg_patternB:
  agg_num_gpu_candidates: [1, 2, 4, 8]
  agg_tp_candidates: [1, 2, 4, 8]
  agg_pp_candidates: [1]
  agg_dp_candidates: [1]
  agg_moe_tp_candidates: [1, 2, 4, 8]
  agg_moe_ep_candidates: [1]
  agg_cp_candidates: [1]
```

Disaggregate mode uses the following exact role maps:

```yaml
prefill_patternA:
  prefill_num_gpu_candidates: [2, 4, 8, 16, 32]
  prefill_tp_candidates: [1, 2, 4, 8]
  prefill_pp_candidates: [1]
  prefill_dp_candidates: [1, 2, 4, 8, 16, 32]
  prefill_moe_tp_candidates: [1]
  prefill_moe_ep_candidates: [2, 4, 8, 16, 32]
  prefill_cp_candidates: [1]

prefill_patternB:
  prefill_num_gpu_candidates: [1, 2, 4, 8]
  prefill_tp_candidates: [1, 2, 4, 8]
  prefill_pp_candidates: [1]
  prefill_dp_candidates: [1]
  prefill_moe_tp_candidates: [1, 2, 4, 8]
  prefill_moe_ep_candidates: [1]
  prefill_cp_candidates: [1]

decode_patternA:
  decode_num_gpu_candidates: [2, 4, 8, 16, 32]
  decode_tp_candidates: [1, 2, 4, 8]
  decode_pp_candidates: [1]
  decode_dp_candidates: [1, 2, 4, 8, 16, 32]
  decode_moe_tp_candidates: [1]
  decode_moe_ep_candidates: [2, 4, 8, 16, 32]
  decode_cp_candidates: [1]

decode_patternB:
  decode_num_gpu_candidates: [1, 2, 4, 8]
  decode_tp_candidates: [1, 2, 4, 8]
  decode_pp_candidates: [1]
  decode_dp_candidates: [1]
  decode_moe_tp_candidates: [1, 2, 4, 8]
  decode_moe_ep_candidates: [1]
  decode_cp_candidates: [1]
```

Expand those maps into four separate disaggregate Task YAMLs:

| Experiment | Prefill role map | Decode role map | Base parallel pairs before batch/worker enumeration |
|---|---|---|---:|
| `disagg_AA` | Pattern A (`17`) | Pattern A (`17`) | `289` |
| `disagg_AB` | Pattern A (`17`) | Pattern B (`4`) | `68` |
| `disagg_BA` | Pattern B (`4`) | Pattern A (`17`) | `68` |
| `disagg_BB` | Pattern B (`4`) | Pattern B (`4`) | `16` |

The disaggregate base Cartesian product is `289+68+68+16=441` prefill/decode parallel pairs before batch-size and worker-count enumeration.

#### Common Task-level fields

The role-map snippets above intentionally show only the parallel candidate axes. Every generated aggregate and disaggregate Task must also set the complete workload/search contract from the matrix table: `database_mode=SOL`, `total_gpus=64`, the selected `isl/osl/ttft`, `prefix=0`, `nextn=0`, `tpot=50000`, `pareto_sweep=false`, and `batch_sweep_step=1`. Aggregate Tasks set `enable_chunked_prefill=false`; disaggregate Tasks set `prefill_enable_chunked_prefill=false`. Both modes must also apply all five neutral correction/scaling values listed below. A role-map snippet is therefore not a complete runnable Task by itself.

#### Worker/replica orchestration

Every disaggregate Task YAML must explicitly set:

```yaml
num_gpu_per_replica: [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
max_gpu_per_replica: 64
max_prefill_workers: 64
max_decode_workers: 64
```

This explicit effective list matches the reference default after applying the `64`-GPU ceiling and avoids relying on a hidden default that also contains values above `64`. `prefill/decode worker_gpus` describes each worker shape; `max_prefill_workers` and `max_decode_workers` bound deployment worker counts; `num_gpu_per_replica` bounds aggregate replica allocation. All remain subject to the overall `total_gpus=64` budget. Result artifacts must record the selected worker shape, prefill/decode worker counts, replica allocation, and final `num_total_gpus` separately.

#### Topology-aware communication semantics

- Preserve the user-requested canonical system `gb300`; do not alias it to the reference TODO's distinct `b300_sxm` system.
- On `h100_sxm`, `h200_sxm`, and simulated `h800_sxm`, communication groups of size `<=8` use `intra_node_bw`; groups `>8` use `inter_node_bw`.
- On `gb300`, groups `<=4` use `intra_node_bw`; groups `5..32` cross physical nodes but remain inside the 72-GPU NVLink rack and use `inter_node_bw=900 GB/s`. The accepted space never reaches `inter_rack_bw`.
- `tp=8` therefore crosses a physical node boundary on `gb300`; the plan must not claim that `tp<=8` is always single-node. It remains inside the rack-level NVLink domain.
- Validation must record the effective bandwidth tier selected for each communication group instead of inferring it only from `tp<=8`.

#### OOM and infeasibility handling

Do not pre-delete parallel rows because they are expected to run out of memory. Memory feasibility is adjudicated by AIC for each model/system/workload/runtime combination. Small workers becoming OOM or memory-infeasible at long ISL is a valid terminal result and must be counted and reported. Unknown errors must fail fast and must not be relabeled as OOM.

Explicitly exclude Step4 `EP=64`, DeepSeek-V4-Pro `EP=64`, a DS-only `EP=64` supplement, model-specific primary spaces, and any generic uneven-EP SOL change from this task. The exclusion is required because current Step4 SOL floors `352/64` to `5` resident experts per rank instead of representing the runtime's `32x6 + 32x5` placement; the global-equivalent omission is `9.091%` and the heavy-rank undercount is `16.667%`.

The reference TODO also contains `isl=524288`, uses `osl=1024` at every ISL, and targets `b300_sxm`. These are deliberate differences, not omissions: this task follows the later user decisions of five binary long-ISL points without `524288`, `osl=1` plus a separate `4K/1024` decode smoke, and canonical `gb300`.

#### Strict roofline-only correction policy

`DatabaseMode.SOL` alone is insufficient because Task defaults can still apply empirical multipliers after the underlying roofline op estimate. For every primary comparison Task, explicitly set the same neutral values for both models:

```yaml
prefill_latency_correction: 1.0
decode_latency_correction: 1.0
rate_match_prefill_degradation: 1.0
rate_match_decode_degradation: 1.0
autoscale_ttft_correction_factor: 1.0
```

The first two defaults otherwise scale every operation in the disaggregate latency breakdown; the rate-match defaults change disaggregate throughput; and the autoscale default changes TTFT before SLA filtering. Equal non-neutral factors are not sufficient for fairness because SLA pruning and agg/disagg ranking interact nonlinearly with model margins. Primary outputs must be labeled theoretical SOL/roofline estimates rather than production predictions. Any future calibrated-serving run must be separately named, use an explicitly reviewed policy for both models, and never be mixed into the primary roofline-only ranking.

Use exhaustive cap-aware batch search:

1. Start every batch sweep at `1` with `batch_sweep_step=1`.
2. Use initial caps aggregate=`1024`, disaggregate prefill=`16`, and disaggregate decode=`1024`.
3. If the rank-1 result equals an active cap, double only that saturated cap and rerun the affected search.
4. Repeat cap doubling until the rank-1 result is below the cap, the configuration is OOM, or the TTFT SLA is infeasible.
5. Treat every still cap-saturated row as non-final; it cannot enter final ranking.

### Required numeric evidence

For every successful point, record at least:

- model
- system
- backend/version
- `isl`
- `osl` used by the test
- `total_gpus`
- `tp`, `pp`, `dp`, `moe_tp`, `moe_ep`, `cp`
- `worker_gpus=tp*dp`, prefill/decode worker count, replica allocation, and final `num_total_gpus`
- effective communication bandwidth tier for each modeled collective/group
- all five effective correction/scaling values, each expected to equal `1.0`
- TTFT
- TPOT
- request latency
- output throughput (`tokens/s`, `tokens/s/gpu`, `seq/s/gpu` as available)
- primary `OSL=1` input/prefill throughput before and after fixed-64-GPU cluster normalization
- estimated memory / kvcache if emitted
- key per-op latency categories if emitted (`gemm`, `fmha`, `moe`, `comm`, `memory`)
- Step4 vs DS-V4-Pro absolute and relative deltas for primary metrics
- TTFT SLA target and TTFT pass/fail status
- explicit evidence that TPOT was observed but not used as an active SLA filter

### Comparison delta and deterministic ranking contract

- For the five primary `OSL=1` workloads, rank eligible rows within each model/system/workload/TTFT/serving-mode group by cluster-normalized input/prefill token throughput descending. Do not use output `tokens/s/gpu_cluster`, which is structurally zero when no decode interval runs.
- For the independent `ISL=4096, OSL=1024` decode smoke, retain output `tokens/s/gpu_cluster` descending as the ranking metric.
- Break exact throughput ties only by a canonical configuration identity ascending. Do not add TTFT, TPOT, or another undeclared performance objective as a tie-breaker.
- Compute cross-model deltas only for aligned matrix keys where both models produced an eligible model-level rank-1 row. The two selected parallel configurations may differ because each model is optimizing over the same search space; do not require the selected config identities themselves to match.
- Define `absolute_delta = Step4 - DeepSeek-V4-Pro` and, for a nonzero DeepSeek-V4-Pro baseline, `relative_delta = absolute_delta / DeepSeek-V4-Pro`. Record DeepSeek-V4-Pro as the explicit baseline. The only zero-baseline exception is observational TPOT when both models report the physically valid `0.0` at primary disaggregate `OSL=1`: preserve both values, emit `absolute_delta=0.0`, `relative_delta=null`, and `status="zero_baseline_both_zero"`. Every non-TPOT zero baseline and every `Step4 != 0` / `DeepSeek-V4-Pro == 0` case must still fail fast; never emit `inf`, `NaN`, epsilon-scaled, or fabricated finite values.
- Record metric polarity alongside every delta: throughput metrics are higher-is-better, while TTFT/TPOT/request latency are lower-is-better. A positive delta therefore does not universally mean improvement.
- If only one model has an eligible row for an aligned matrix key, retain that model's ranking evidence, mark the comparison key as unpaired, and do not fabricate a cross-model delta.

### Acceptance thresholds

Because this is a model-definition task, the primary acceptance threshold is structural correctness and reproducibility, not matching external silicon numbers.

- All structural tests pass exactly.
- Matrix jobs complete or fail fast with documented unsupported configurations.
- No Step4 result uses profiling-based `perfdb` in the Step4 op latency path.
- DS-V4-Pro comparison is included as reference, with clear note that DS-V4-Pro may have model-specific measured paths depending on current AIC implementation and database mode.
- Every retained row satisfies its requested TTFT SLA; TPOT is not an acceptance filter and must not be described as one.
- Best configurations use the workload-specific declared metric: cluster-normalized input/prefill throughput for the five `OSL=1` primary workloads, and output `tokens/s/gpu_cluster` for the `4K/1024` decode smoke.
- Every final ranked row is non-cap-saturated, and both models were searched over the same exact `21` parallel configurations.
- Pattern A/B materialization is exactly `17/4`, with `8` rows at `EP>8`, `4` rows at `EP=16`, `4` rows at `EP=32`, and maximum `worker_gpus=32`.
- Disaggregate pairing materialization is exactly `AA=289`, `AB=68`, `BA=68`, `BB=16`, total=`441` base prefill/decode pairs before batch/worker enumeration.
- Every disaggregate task exposes worker/replica ceilings `64/64/64`, and every primary task reports all five empirical correction/scaling factors as `1.0`.

## Execution Phases for Later Implementation

### Phase 0: Pre-implementation safety gate

- Verify the checkout still targets the user-confirmed `step-design` branch before any implementation action.
- Resolve pre-existing dirty workspace by user-approved commit/stash/worktree strategy.
- Re-read this task's `requirements.md`, `notes.md`, `issues.md`, and `plan.md`.
- If implementation touches `src/aiconfigurator/generator/**`, read `.claude/rules/generator-development.md` first; current plan does not require it.
- If implementation touches collector paths, read `.agents/skills/aic-collector-op-development/SKILL.md` first; current plan does not require it.

### Phase 1: TDD RED tests

- Add tests that expect Step4 parsing/registration and ops structure.
- Run targeted tests and observe failure because Step4 is not yet implemented.
- Record RED evidence in a test report under this task directory.

### Phase 2: Minimal implementation

- Add local Step4 config and model family wiring.
- Add Step4 model class using generic roofline-capable ops.
- Add fail-fast validation for inconsistent layer counts and unsupported fields.

### Phase 3: GREEN tests

- Run targeted unit tests until all pass.
- Run relevant existing unit tests to prevent regressions.

### Phase 4: Matrix validation

- Run the requested Step4 vs DS-V4-Pro matrix in `SOL` mode with explicit `nextn=0` for both models.
- Run every required workload in both `agg` and `disagg` serving modes; preserve the same model/system/sequence settings across the two modes.
- Run the five TTFT-only SLA targets `200`, `500`, `1000`, `2000`, and `5000` ms with explicit `tpot=50000` and `pareto_sweep=false`; rank the five primary `OSL=1` workloads by cluster-normalized input/prefill token throughput and the `4K/1024` decode smoke by output `tokens/s/gpu_cluster`.
- Set `enable_chunked_prefill=false` for aggregate workloads and `prefill_enable_chunked_prefill=false` for disaggregate workloads; do not add an aggregate-only supplement.
- Use exact input lengths `isl=[4096,16384,65536,262144,1048576]`; the original `iso` spelling is interpreted as `isl`.
- Set explicit `osl=1` in every five-point primary-matrix command and result row.
- Run a separate Step4-vs-DS-V4-Pro decode smoke at `isl=4096, osl=1024`, again with explicit `nextn=0` for both models.
- Set explicit `prefix=0` in every primary-matrix and decode-smoke command; do not add prefix-cache reuse to the required matrix.
- Run the exact Step4-only `nextn=3` structural/smoke point specified above; do not turn it into a cross-model performance row.
- Use the accepted common `21`-configuration Pattern A/B space for both models and every aggregate/prefill/decode search; retain `AA`, `AB`, `BA`, and `BB` pairings for disaggregate mode.
- Materialize Pattern A and Pattern B in separate aggregate Tasks and expand disaggregate mode into separate `AA/AB/BA/BB` Tasks using role-prefixed candidate fields; do not combine the two patterns into one broad candidate bundle.
- Set `num_gpu_per_replica=[1,2,4,8,16,24,32,40,48,56,64]`, `max_gpu_per_replica=64`, `max_prefill_workers=64`, and `max_decode_workers=64`; record worker shape, worker count, replica allocation, and total GPUs as distinct fields.
- Set `prefill_latency_correction=1.0`, `decode_latency_correction=1.0`, `rate_match_prefill_degradation=1.0`, `rate_match_decode_degradation=1.0`, and `autoscale_ttft_correction_factor=1.0` for both models; reject any primary result with a non-neutral effective value.
- Keep all 21 parallel rows until runtime memory adjudication. Record OOM/memory-infeasible rows as terminal outcomes and fail fast on unknown errors.
- Audit communication against each system's topology: GB300 uses a 4-GPU physical-node boundary and a 72-GPU rack boundary; H100/H200/H800 use an 8-GPU node boundary. Record the effective bandwidth tier.
- Exclude `EP=64` for both models and do not add a DS-only supplement or generic uneven-EP SOL modification.
- Start batch sweeps at `1` with step `1`; initialize caps at aggregate=`1024`, prefill=`16`, and decode=`1024`; double only saturated caps and rerun until non-saturated, OOM, or TTFT-SLA-infeasible. Never rank a cap-saturated row as final.
- Include H800 through `--systems-paths default,tests/performance/aic_roofline_pareto/systems`; label all H800 rows `simulation_status=simulated` and `database_mode=SOL`.
- Do not add `h800_sxm` to built-in systems or claim H800 silicon/perfdb validation in this task.
- Save numeric outputs and create a Markdown test report in this task directory.

### Phase 5: Review and finalization

- Run independent review gate for architecture/modeling decisions.
- Fix any review findings with tests.
- Update `summary.md`, `lessons.md`, and final test reports.

### Phase 6: Branch review, archival commit, and `step-design` integration

- Audit every tracked modification and untracked candidate in `task/step4-predefined-ops`; classify source, tests, durable task evidence, disposable runtime state, and generated outputs before staging.
- Compare overlapping local changes in the sibling `step-design` worktree byte-for-byte and by Git diff so no independent work is overwritten or silently duplicated.
- Run the required independent code-quality and architecture review lanes over the complete branch diff, then obtain the mandatory StepCode Claude cross-model review. Resolve every `CRITICAL`, `HIGH`, `BLOCK`, or `REQUEST CHANGES` finding before commit.
- Run focused tests, the full unit suite, Ruff lint/format, `git diff --check`, task-document validation, and archive-hash verification from the current branch state.
- Refresh `progress.md`, `issues.md`, `review.md`, `summary.md`, `lessons.md`, and the test report with review, commit, push, merge, and verification evidence. Keep raw user intent only in `requirements.md`.
- Stage only reviewed deliverables. Exclude disposable coverage/runtime caches and other non-deliverable outputs. Commit using the repository Lore protocol and verify the exact committed file inventory.
- Push `task/step4-predefined-ops` to `origin`, verify the remote ref, then reconcile the sibling `step-design` worktree's pre-existing changes without loss.
- Merge the feature branch into `step-design`, rerun the required validation on the merged tree, push `step-design`, and verify both remote refs point to the intended commits.
- Do not remove either host-owned worktree or delete any branch as part of this task.

Phase 6 acceptance criteria:

1. No unreviewed source, test, or documentation change enters the commit.
2. No disposable `.coverage`, `.omc/`, cache, or unrelated output artifact enters the commit.
3. The feature-branch commit is reproducible, tests/static checks pass, and `origin/task/step4-predefined-ops` matches the local feature HEAD.
4. The target worktree's pre-existing modifications are either proven identical and safely absorbed or preserved in a separate auditable commit; none is overwritten.
5. The merged `step-design` tree passes the required validation and `origin/step-design` matches the verified merged HEAD.
6. Final task docs contain exact commit hashes, remote-ref evidence, test metrics, deliverable hashes, and zero unresolved merge blockers.

## Current Execution Gate

The docs/grilling stage, user-authorized implementation, remediation, regression, representative artifacts, full matrix, strict merge, merged semantic validation, final documentation, static verification, and independent completion review are complete. The following matrix-entry conditions were all satisfied before execution:

1. ISSUE-048 has an explicit aggregate attention semantic contract covered by observed RED/GREEN tests.
2. ISSUE-049 strictly binds every normalized published row to the final successful experiment evidence with an exact schema and mutation coverage at build, commit, load, merge, and finalize boundaries.
3. ISSUE-050 removes executed-attention provenance fallback and distinguishes executed attention from the explicit `gen_tokens == 0` no-op placeholder.
4. ISSUE-051 enforces an exact checkpoint-header schema and initialization atomicity without malformed-checkpoint recovery.
5. ISSUE-052 enforces exact positive-`int` validation for `batch_sweep_step` at both public Task and sweep-helper boundaries.
6. Aggregate and disaggregate sweeps propagate every unknown per-config exception instead of publishing a partial 21-config search result; known typed memory/SLA terminal outcomes remain explicit.
7. Every public aggregate or disaggregate `Task` containing Step4 rejects `None`, `SILICON`, `HYBRID`, and `EMPIRICAL` and admits only formula-only `SOL` or `SOL_FULL` before any perfdb load.
8. Collective bandwidth evidence comes from the actual communication query group and selected bandwidth, not a second derivation from normalized parallel dimensions.
9. Final paired model comparisons include TPOT values, absolute delta, relative delta, lower-is-better polarity, and an explicit delta status even though TPOT remains non-binding for eligibility. The legal primary-disaggregate `0.0/0.0` TPOT case publishes `relative_delta=null`; all other zero-baseline cases fail fast.
10. Every matrix point pins `engine_step_backend="python"`; Task propagation, mode-run identity, checkpoint matrix hash, and execution-contract fingerprint all bind that value.
11. H200 primary/decode and real `2+2` shard artifacts have been regenerated under the new execution fingerprint; all stale pre-fix artifacts remain excluded from ranking.
12. Independent code and architecture reviews return no `BLOCK` or unresolved `REQUEST CHANGES`.
13. Focused regression, full unit tests, Ruff, formatting, and static checks pass with fresh evidence.
14. Only then may the complete `480` mode-run matrix start.

Current status snapshot:

- Gates 1-5: complete. Focused remediation, broader regression, fresh representative artifacts, strict resume/shard verification, independent post-change review, and post-change full-unit regression all passed.
- Gate 6 / ISSUE-053: strict RED/GREEN complete. Unknown aggregate and disaggregate per-configuration exceptions now propagate without partial-result publication; focused=`28/28`, related regression=`109/109`.
- Gate 7 / ISSUE-054: strict RED/GREEN complete. Step4 admits only `SOL` and `SOL_FULL` at construction, validation, and `run(validate=False)` boundaries before any perfdb load; the same focused and related suites passed.
- Gate 8 / ISSUE-055: final independent StepCode Claude verdict=`APPROVE CONTEXT_CAPTURE`; strict RED/GREEN remediation, representative verification, and independent post-change review are complete. Outer public query wrappers capture immutable ordered `CommunicationQueryEvidence`; cache hits and duplicate queries remain observable; operation identity is token-scoped from the executing `Operation._name`; `None` and requested-empty `()` remain distinct; `PerformanceResult` is not an evidence carrier; exact selected-point reruns own capture; checkpoint schema=`3` rejects older checkpoints; runner-side `_bandwidth_evidence()` reconstruction is removed.
- Gate 9 / ISSUE-056 and ISSUE-059: strict RED/GREEN remediation and representative verification are complete. Final JSON/CSV/Markdown comparisons publish TPOT values, absolute/relative deltas, `lower_is_better` polarity, and an explicit status while `tpot_observed_only=True` keeps eligibility TTFT-only. The legal TPOT `0.0/0.0` path emits `relative_delta=null`; every other zero baseline still fails fast. Targeted/performance/combined tests passed `125/125`, `162/162`, and `419/419`; the independent post-change review returned `APPROVE`.
- Gate 10 / ISSUE-057: strict RED/GREEN remediation is complete. Every matrix point fixes `engine_step_backend="python"`; Task/RuntimeConfig propagation, normalized rows, mode identity, matrix hash, execution contract, resume compatibility, and final artifacts bind the value. Targeted GREEN passed `12/12`.
- Leader-owned post-format combined regression passed `413/413` in `52.67s`; performance runner/checkpoint/CLI passed `156/156`, SDK/capture passed `257/257`, and the Gate 6/7 regression passed `109/109` after the final docstring correction. Ruff check, Ruff format, and `git diff --check` pass over all `33` changed Python files.
- The first post-Gate-10 H200 primary v1 run was intentionally interrupted after `1` committed mode run because a stale `sweep_agg()` `Raises` docstring would have changed the SDK source fingerprint after completion. The partial directory is immutable interruption evidence and must not be resumed or ranked. Final H200 primary/decode and real `2+2` shard artifacts must use new directories after the correction.
- The H200 primary v2 run completed and durably stored all `4/4` mode-run records, but finalization failed with `ValueError: Cannot compute tpot: zero DeepSeek-V4-Pro baseline`. Primary disaggregate `OSL=1` produced the legitimate physical result `Step4 TPOT=0.0` and `DeepSeek-V4-Pro TPOT=0.0`; only `mode_runs.sqlite3` exists, so v2 is immutable failure evidence and is not a successful representative artifact.
- ISSUE-059 independent StepCode Claude verdict=`APPROVE`; artifact SHA256=`6f2c530e32a3b9fe309d5ff8ac5a15b0b7cc590ea64fa979f5082e93f95c01bc`; material scope expansion=`no`. The stricter accepted contract permits `relative_delta=null` only for TPOT with both values exactly zero and requires an explicit status. Strict RED/GREEN, post-format combined regression=`419/419`, Ruff, formatting, and diff checks now pass. Fresh H200 primary evidence must use a new v3 directory.
- Fresh H200 primary v3 completed `4` mode runs, `12` successful experiment terminals, `12` normalized/ranked rows, and `2` paired comparisons. Disaggregate TPOT publishes `0.0/0.0`, `absolute_delta=0.0`, `relative_delta=null`, and `status="zero_baseline_both_zero"`; aggregate TPOT remains numeric. Same-directory strict resume executed `0` mode runs, retained `4` SQLite rows, and preserved all five artifact hashes.
- Fresh H200 decode v2 completed the same `4/12/12/2` counts. Aggregate TPOT Step4/DeepSeek=`91.0494843644669/122.49266142144401 ms`; disaggregate=`37.305/33.362 ms`; every delta status is `computed`. The real Step4/DeepSeek shards contain `2+2` records, resume with zero loader/executor calls, merge to `4` rows, preserve the merged checkpoint hash across resume, and reproduce all four final artifacts byte-for-byte.
- Representative Gate 11 and independent READ-ONLY Gate 12 are complete. Gate 12 verdict=`APPROVE`; artifact SHA256=`9d7eb4468e1e76be3dfc051639c53fc58d7c97e770d55216d2e435e0df23d5d3`. Four non-blocking WATCH observations are tracked in ISSUE-060 and Checkpoint 63.
- Gate 13 post-change full-unit regression is complete: collected=`3111`, selected=`1992`, passed=`1986`, skipped=`12`, deselected=`1119`, warnings=`4`, elapsed=`829.31s`, exit code=`0`. Log SHA256=`068918e1c6d87b80c5f6187ba1725adee905e16e203b025cc666cc343b88b04e`.
- Gate 14 completed the frozen `480` mode-run matrix through four fresh system-shard directories under the approved maximum concurrency of three; the search space and all accepted matrix contracts remained unchanged.
- Gate 14 sharding must not use the filtered `--system` CLI contract hash. Build the complete `480`-spec execution contract once, require full contract SHA256 `63aa70620a6fa8908bf0747390d3833756af6b88c0a11aecb21e63cd81da8297` and git HEAD `9ce84ebbe3a0d7f785c91d055bdbdf4fdaabcbf1`, then execute each exact `120`-spec system shard through the existing `execute_matrix_runs()` API with that shared hash. Merge only through `merge_completed_checkpoints()` after all four shards are complete. Independent review verdict=`APPROVE`; artifact SHA256=`cd9a9710d0f46a2aa2d15c5525ebdbb96175dbd9db3eeefd37bfbde2c334e223`.
- Gate 14 Wave 1 is complete. GB300, H200, and H100 each completed `120/120` mode runs with normalized/ranked rows=`197/197`, `162/162`, and `98/98`; all exited `0` and strict-resumed with loader/executor calls=`0/0` plus hashes unchanged=`5/5`. Wave 2 simulated-SOL H800 was then released as the only active matrix lane.
- Gate 14 Wave 2 is complete. H800 completed `120/120` mode runs with normalized/ranked rows=`88/88`, exit code `0`, exact row-label sets `simulation_status={simulated}` and `database_mode={SOL}`, and strict resume loader/executor calls=`0/0` with hashes unchanged=`5/5`. All four shards cover `480/480` mode runs, and the strict full merge subsequently completed.
- Gate 14 strict merge is complete through the existing `merge_completed_checkpoints()` API: mode runs=`480`, normalized/ranked rows=`545/545`, paired/unpaired comparisons=`89/28`, H800 rows=`88`, and merge exit code=`0`. Merged strict resume returned records=`480`, loader/executor calls=`0/0`, hashes unchanged=`5/5`, and exit code=`0`.
- Merged semantic validation passed after correcting three disposable-validator schema assumptions without changing artifacts: `ParallelRow.worker_gpus` is a property rather than `num_gpu`; DeepSeek uses an empty approximation-group mapping; and aggregate `per_ops_data` includes scheduling metadata while six explicit zero-latency generation-attention placeholders use source `not_executed`. Final counts are SOL op entries=`20280`, explicit no-op entries=`6`, success/memory/SLA terminals=`545/655/240`, Step4/DeepSeek normalized rows=`348/197`, paired/unpaired=`89/28`, TPOT computed/zero-zero=`59/30`, and semantic exit code=`0`.
- Numeric comparison evidence is published in `comparison_summary.md` with authoritative raw per-key source `result-full-matrix-v2-merged-20260715/model_comparisons.csv`. The corrected-v2 comparison table is numerically identical to the superseded v1 table and binds source JSON SHA256 `d18e05806968518ef5f46206f77a29eda76c72e140d8c7502b2c4305790f0ed8`. It aggregates paired Step4/DeepSeek values and deltas by system, workload, primary ISL, TTFT SLA, serving mode, and metric; H800 is labeled simulated SOL and long-context Step4 rows retain the temporary-MLA limitation.
- Final static validation passed after removing six Markdown hard-break trailing spaces from the new test report: Ruff lint exit=`0`, Ruff format exit=`0` with `426` files already formatted, `git diff --check` exit=`0`, and explicit trailing-whitespace lines=`0` across `10` task docs.
- Final independent completion review returned leading verdict=`APPROVE`, required actions=`none`, artifact SHA256=`77bbe32d1d128c5457a8440915d1777b27176e0c1a2ec54129f2ed9436187d86`. The reviewer retained non-blocking future observations for long-context feasibility and narrow CI integration coverage. Its separate 21-row-test observation was closed by direct evidence from `test_common_vllm_parallel_rows_match_authoritative_21_row_space`, which already asserts the authoritative 21-row set. The v1 matrix gates completed at this checkpoint; Phase 6 corrected-v2 archival and integration gates remain governed by the current integration section below.
- Final completion validation rechecked the five merged hashes, SQLite header/`480` rows, contract/matrix/HEAD identity, all semantic counts, `51` summary-inventory entries, final review verdict, full-unit log hash/summary, static log, and ten-document whitespace state. The first disposable run failed only because its raw-log substring assertion did not remove ANSI color escapes; the corrected full rerun passed with exit code `0` and no repository or result change.

- Formatter-consistent corrected-v3 evidence supersedes the corrected-v2 archive for delivery. All four exact `120`-record shards use execution contract `a13a4fe6ef9b932d01772ee3f0b8844760c52ec991fdcc5af641186baf1b697c`; strict merge produced `480` records, strict resume executed loader/executor=`0/0`, and all five hashes remained unchanged. The merged semantic audit retained normalized/ranked=`545/545`, paired/unpaired=`89/28`, terminals=`545/655/240`, SOL/no-op=`20280/6`, nested generation evidence=`2935/4338/1403`, and outer `generation_moe_overlap=0`. v2/v3 scientific values are identical after normalizing only execution-contract provenance.
- The formatter-consistent full unit run is the final full-suite evidence: collected=`3127`, selected=`2008`, passed=`2002`, skipped=`12`, deselected=`1119`, warnings=`4`, elapsed=`765.21s`, exit=`0`. Per the user's minimum-cost delivery instruction, it must not be rerun unless source or tests change.

## Current Grilling Order

None. Grilling decisions and the final independent architecture/plan review are complete.

## Current Integration Gate

Phase 6 is in progress. Corrected-v3 GB300 completed `120/120`; all four exact system shards merged to `480` records under formatter-consistent execution contract `a13a4fe6ef9b932d01772ee3f0b8844760c52ec991fdcc5af641186baf1b697c`. Strict resume returned loader/executor=`0/0` with five unchanged hashes, merged semantic validation passed, and `comparison_summary.md` now binds merged-v3. The fresh formatter-consistent full-unit evidence is `2002 passed, 12 skipped, 0 failed`; focused/integration=`178/178`; Ruff, format, diff, primary `49+1` inventory, and figure `41` inventory gates pass. Final independent StepCode Claude verdict=`APPROVE`, previous stale-contract blocker resolved=`YES`, CRITICAL/HIGH/MEDIUM=`0/0/0`, required actions=`none`. Only final exact inventory refresh, exact staging, Lore commit, feature push, exact 16-path target preservation stash, `ff-only` merge, minimal merged-tree verification, and target push remain, in that order.
