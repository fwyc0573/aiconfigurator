## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-16 | Created the final Step4-Pro-V1 test report with TDD history, full regression, static checks, direct rooflines, SDK/CLI metrics, environment diagnostics, and independent-review evidence. |
| 2026-07-16 | Added validation evidence for the standalone roofline-model review document and independent Claude approval. |

# Test Report: Step4-Pro-V1 AIC Support

**Date:** 2026-07-16
**Overall Result:** PASS
**Worktree:** `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro`
**Branch:** `step4-pro`
**Reviewed Code Range:** `fdd869b..e4a1083`

## 1. Test Script Information

### Environment

| Item | Actual Value |
|---|---|
| Conda environment | `/home/i-fengyicheng/miniconda3/envs/aic-step-design` |
| Python | `3.11.15` |
| pytest | `8.4.2` |
| Ruff | `0.14.1` |
| Source binding | `PYTHONPATH="$PWD/src:$PWD"` |
| Headless plotting | `MPLBACKEND=Agg` |
| Final full-unit temp directory | `TMPDIR=/tmp` after byte/inode preflight |
| Independently verified short data temp directory | `/data/ycfeng/tmp` |

The repository `.venv` was not used for validation because it contained Python `3.13.13`, lacked pytest, and had no Ruff executable. The first `.venv` baseline exited before collection with `ModuleNotFoundError: pytest`; this was an environment failure, not product evidence.

### Authoritative Input

| Input | Expected | Actual | Result |
|---|---:|---:|---|
| CSV path | `/data/ycfeng/stepfun-performance-optimization/permormancebenchmark/architecture_calculator_v1 - Main - latest.csv` | Same | PASS |
| CSV bytes | `6423` | `6423` | PASS |
| CSV SHA256 | `f76fca03fd10eb145a04ff9ed906cdbd52beee37609103f9c99006e2bbf1920b` | Same | PASS |

### Test Files

- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/tests/unit/sdk/models/test_step4_pro_v1.py`
- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/tests/unit/sdk/database/test_step4_pro_v1_roofline.py`
- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/tests/integration/test_step4_pro_v1_support.py`
- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/tests/unit/sdk/models/test_step4.py`
- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/tests/unit/sdk/database/test_step4_roofline.py`
- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/tests/unit/sdk/database/test_collective_query_capture.py`
- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/tests/unit/sdk/database/test_base_queries.py`
- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/tests/integration/test_step4_prefill_ranking.py`
- `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/tests/unit/collector/test_parallel_run.py`

### Reproducible Commands

Common environment:

```bash
cd /data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro
PY=/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python
export PYTHONPATH="$PWD/src:$PWD"
export MPLBACKEND=Agg
```

Focused Step4-Pro-V1 and public integration:

```bash
TMPDIR=/tmp "$PY" -m pytest -p no:cacheprovider -q \
  tests/unit/sdk/models/test_step4_pro_v1.py \
  tests/unit/sdk/database/test_step4_pro_v1_roofline.py \
  tests/integration/test_step4_pro_v1_support.py
```

Original Step4 and shared regression:

```bash
TMPDIR=/tmp "$PY" -m pytest -p no:cacheprovider -q \
  tests/unit/sdk/models/test_step4.py \
  tests/unit/sdk/database/test_step4_roofline.py \
  tests/unit/sdk/database/test_collective_query_capture.py \
  tests/unit/sdk/database/test_base_queries.py \
  tests/integration/test_step4_prefill_ranking.py
```

Full unit regression:

```bash
export TMPDIR=/tmp
"$PY" -m pytest -p no:cacheprovider -q -m unit
```

AF_UNIX controlled reproduction and resolution:

```bash
TMPDIR="$PWD/tests/.tmp" "$PY" -m pytest -p no:cacheprovider -q \
  tests/unit/collector/test_parallel_run.py::TestNormalCompletion::test_single_worker
TMPDIR=/tmp "$PY" -m pytest -p no:cacheprovider -q \
  tests/unit/collector/test_parallel_run.py::TestNormalCompletion::test_single_worker
TMPDIR=/tmp "$PY" -m pytest -p no:cacheprovider -q \
  tests/unit/collector/test_parallel_run.py
TMPDIR=/data/ycfeng/tmp "$PY" -m pytest -p no:cacheprovider -q \
  tests/unit/collector/test_parallel_run.py
```

Static checks:

```bash
"$PY" -m ruff check .
"$PY" -m ruff format --check .
"$PY" -m ruff format --check --exclude tests/.tmp .
git ls-files -z -- '*.py' | xargs -0 "$PY" -m ruff format --check
git diff --check fdd869b..HEAD
```

Independent review:

```bash
omx ask claude "<final review prompt covering fdd869b..e4a1083 and all required artifacts>"
```

The exact prompt and raw reviewer output are preserved in `.omx/artifacts/claude-act-as-the-independent-final-code-reviewer-for-the-step4-pro-2026-07-15T19-36-24-039Z.md`.

## 2. Validation Criteria

- The exact cached identity `stepfun-ai/Step4-Pro-V1` resolves offline.
- CSV-provided topology and integer parameter arithmetic close exactly, including `4 dense + 76 MoE = 80` and `20 Full + 60 SWA = 80`.
- Malformed Step4 routed-MoE/core geometry and non-divisible TP/EP layouts fail fast with field-specific errors.
- Original Step4 derived widths remain exactly `2112`, `24576`, and `32768`.
- The complete operation graph executes with `DatabaseMode.SOL`; every executed operation has `source="sol"` and no profile loader or network path is used.
- Every direct `SOL_FULL` tuple satisfies `selected == max(math, memory)` and equals scalar `SOL` for the same query.
- Aggregate and disaggregate public SDK flows return finite, positive TTFT/TPOT and preserve per-operation provenance.
- Offline CLI estimate and generate resolve the cached model; generate is treated only as artifact smoke and must expose its no-fit warning.
- Original Step4/shared tests and the full unit suite have zero failures.
- All delivery Python files pass Ruff lint and format checks; committed diffs have no whitespace errors.
- Documentation states the attention, KV-cache, AFD, Task-level SOL_FULL, and generic naive-generator limitations without scaling or fallback.
- Independent StepCode Claude review returns no BLOCK before archival.

## 3. Test Results and Evidence

### Result Matrix

| Suite / Check | Result | Actual Counts / Values | Elapsed | Exit |
|---|---|---|---:|---:|
| Pre-change Step4 baseline | PASS | 90 passed / 90 collected | 7.85 s | 0 |
| Final focused Step4-Pro-V1 | PASS | 65 passed | 10.56 s | 0 |
| Original Step4/shared regression | PASS | 132 passed | 14.45 s | 0 |
| Offline integration warning regression | PASS | 4 passed | 3.20 s | 0 |
| Affected graph/database regression checkpoint | PASS | 192 passed | 18.67 s | 0 |
| Initial full unit with long TMPDIR | EXPECTED ENV FAILURE | 2050 passed / 13 failed / 12 skipped / 1123 deselected / 4 warnings | 800.15 s | 1 |
| Long-TMPDIR single-test control | EXPECTED ENV FAILURE | 1 failed | 0.36 s | 1 |
| Short `/tmp` single-test experiment | PASS | 1 passed | 0.24 s | 0 |
| Short `/tmp` collector suite | PASS | 22 passed | 6.00 s | 0 |
| Short `/data/ycfeng/tmp` collector suite | PASS | 22 passed | 7.18 s | 0 |
| Final full unit with `/tmp` | PASS | 2063 passed / 12 skipped / 1123 deselected / 4 warnings | 770.74 s | 0 |
| Ruff lint | PASS | All checks passed | < 1 s | 0 |
| Full-tree Ruff format diagnostic | EXPECTED TEMP-SCOPE FAILURE | 4 generated fixture copies reported / 434 files formatted | < 1 s | 1 |
| Git-tracked Python format | PASS | 432 tracked Python files formatted | < 1 s | 0 |
| Ruff format excluding `tests/.tmp` | PASS | Delivery tree formatted | < 1 s | 0 |
| Git diff check | PASS | 0 whitespace errors | < 1 s | 0 |
| Independent StepCode Claude review | APPROVE | 0 Critical / 0 BLOCK / 2 low-severity Important / 3 Minor | 240.62 s | 0 |

Counts in this table overlap by design; they are evidence for different checkpoints and must not be summed.

### TDD RED -> GREEN Evidence

| Behavior | RED Evidence | Root Cause | GREEN / Regression Evidence |
|---|---|---|---|
| Cached identity and topology | `4 failed / 1 deselected`, 0.27 s | Model/config absent; first config draft had 81 rather than 80 blocks. | `5 passed`, 0.03 s after exact `4 + 20 + 56` correction. |
| Fail-fast Step4 parser | `18/18` new invalid-input cases failed, 0.39 s | Generic extraction silently substituted or accepted malformed routed-MoE values. | `75/75` Pro/original model tests passed, 0.12 s. |
| Config-derived temporary-MLA geometry | Expected `1312`, actual frozen `2112`; `1 failed / 23 deselected`, 0.18 s | Three Step4 projection widths were hard-coded for one geometry. | `76/76` passed, 0.11 s; original Step4 remains `2112/24576/32768`. |
| Parallel divisibility | `6 failed / 24 deselected`, 0.22 s | Five integer divisions truncated silently; head divisibility used a removable shared assertion. | `6 passed / 24 deselected`, 0.04 s; combined `82/82` passed, 0.13 s. |
| CustomAllReduce SOL_FULL | `26 passed / 1 failed`, 7.53 s; tuple was `(0.44040192, 0, 0)` | Bandwidth-derived transfer time was not classified as memory roofline. | Targeted `2/2` passed, 4.04 s; database regression `106/106` passed, 18.03 s. |
| Structural graph helper | `5 failed / 108 passed`, 7.71 s | Test imported non-exported `models.ops` rather than the public operations module. | `113/113` passed, 7.50 s; affected regression `192/192` passed. |
| Public integration phases | `1 failed / 3 passed`, 3.43 s | Test omitted `genonly_step`; second iteration exposed intentional `not_executed` metadata. | `4/4` passed, 3.23 s; all executed sources constrained to `sol`. |
| Numeric evidence script | `KeyError: scheduling` before report generation | `scheduling` is aggregate metadata and intentionally has no operation-source map. | Corrected evidence code validates operation phases and records scheduling separately; no production change. |

### Exact Architecture and Parameter Closure

| Metric | Expected / CSV | Actual Model Arithmetic | Absolute Delta | Result |
|---|---:|---:|---:|---|
| Hidden size | `6,144` | `6,144` | `0` | PASS |
| Layers | `80` | `80` | `0` | PASS |
| Full layers | `20` | `20` | `0` | PASS |
| SWA / non-Full layers | `60` | `60` | `0` | PASS |
| Dense layers | `4` | `4` | `0` | PASS |
| MoE layers | `76` | `76` | `0` | PASS |
| Dense intermediate | `16,384` | `16,384` | `0` | PASS |
| Routed experts | `512` | `512` | `0` | PASS |
| Top-k | `8` | `8` | `0` | PASS |
| Routed expert intermediate | `2,048` | `2,048` | `0` | PASS |
| Shared expert intermediate | `2,048` | `2,048` | `0` | PASS |
| Vocabulary | `128,896` | `128,896` | `0` | PASS |
| Dense parameters/layer | `301,989,888` | `301,989,888` | `0` | PASS |
| Router parameters/layer | `3,145,728` | `3,145,728` | `0` | PASS |
| Active MoE parameters/layer | `342,884,352` | `342,884,352` | `0` | PASS |
| All MoE parameters/layer | `19,368,247,296` | `19,368,247,296` | `0` | PASS |
| Attention total | `15,896,603,520` | `15,896,603,520` | `0` | PASS |
| RMS total | `983,040` | `983,040` | `0` | PASS |
| Total without embedding | `1,489,092,340,608` | `1,489,092,340,608` | `0` | PASS |
| Embedding total | `1,583,874,048` | `1,583,874,048` | `0` | PASS |
| Total with embedding | `1,490,676,214,656` | `1,490,676,214,656` | `0` | PASS |
| Total activation parameters | `43,164,756,864` | `43,164,756,864` | `0` | PASS |

Exact block order: `("dense_swa",) * 4 + ("moe_full",) * 20 + ("moe_swa",) * 56`.

### Attention and KV-Cache Discrepancy Evidence

| Item | Authoritative / Expected | Temporary Step4-derived Actual | Absolute Gap | Relative Gap / Ratio |
|---|---:|---:|---:|---:|
| Full attention parameters/layer | `153,095,232` | `113,246,208` | `39,849,024` | `26.0289125137%` below CSV |
| SWA attention parameters/layer | `213,911,648` | `163,577,856` | `50,333,792` | `23.5301782164%` below CSV |
| KV cache at 1,048,576 FP8 tokens | `10.7 GB` | `48.31838208 GB` | `37.61838208 GB` | `4.515736642991x`; `351.5736642991%` over target |

Acceptance decision: these non-closing values remain explicit human-update items. No projection was fabricated and no scaling factor was applied.

### Direct SOL_FULL Roofline Audit

| Query Family | Selected ms | Math ms | Memory ms | Scalar SOL ms | Source | `selected=max` | `selected=scalar` |
|---|---:|---:|---:|---:|---|---|---|
| `context_mla` | `171.79869184` | `171.79869184` | `0.12582912` | `171.79869184` | `sol` | TRUE | TRUE |
| `custom_allreduce` | `0.44040192` | `0` | `0.44040192` | `0.44040192` | `sol` | TRUE | TRUE |
| `gemm` | `154.618822656` | `154.618822656` | `0.088080384` | `154.618822656` | `sol` | TRUE | TRUE |
| `generation_mla` | `0.142606336` | `0.142606336` | `0.004787072` | `0.142606336` | `sol` | TRUE | TRUE |
| `mem_op` | `0.050331648` | `0` | `0.050331648` | `0.050331648` | `sol` | TRUE | TRUE |
| `mla_bmm` | `0.002097152` | `0.002097152` | `0.001069056` | `0.002097152` | `sol` | TRUE | TRUE |
| `moe` | `154.618822656` | `154.618822656` | `2.843738112` | `154.618822656` | `sol` | TRUE | TRUE |
| `nccl` | `3.7748736` | `0` | `3.7748736` | `3.7748736` | `sol` | TRUE | TRUE |
| `p2p` | `0.50331648` | `0` | `0.50331648` | `0.50331648` | `sol` | TRUE | TRUE |

Observed selected-latency range: `0.002097152 ms` (MLA BMM) to `171.79869184 ms` (context MLA). All nine families satisfy both invariants.

### Public SDK Metrics

| Configuration | Actual |
|---|---|
| `tp` | `8` |
| `pp` | `2` |
| `dp` | `1` |
| `attention_dp` | `1` |
| `moe_tp` | `8` |
| `moe_ep` | `1` |
| `batch_size` | `1` |
| `ctx_tokens` | `128` |
| `isl` | `128` |
| `osl` | `2` |
| `nextn` | `0` |

| Mode | TTFT ms | TPOT ms | Phase | Operations | Source Counts | Executed Sum ms | Min ms | Max ms |
|---|---:|---:|---|---:|---|---:|---:|---:|
| aggregate | `105.16356682` | `1.32416116666667` | `genonly_step` | `28` | `sol=28` | `1.32416116666667` | `2.56e-06` | `0.70279936` |
| aggregate | `105.16356682` | `1.32416116666667` | `mix_step` | `34` | `not_executed=1, sol=33` | `41.16356682` | `0` | `38.46373376` |
| disaggregate | `45.28` | `1.43` | `decode` | `28` | `sol=28` | `1.430083692` | `2.7648e-06` | `0.7590233088` |
| disaggregate | `45.28` | `1.43` | `prefill` | `33` | `sol=33` | `45.279923502` | `0` | `42.310107136` |

Aggregate scheduling metadata is intentionally separate from operation provenance:

| Metadata | Actual ms / Count |
|---|---:|
| `genonly_step_latency_ms` | `1.32416116666667` |
| `mix_step_latency_ms` | `41.16356682` |
| `num_genonly_steps` | `1` |
| `num_mix_steps` | `1` |

### Per-Operation Source and Latency Evidence

#### Aggregate `genonly_step`

| Operation | Latency ms | Source |
|---|---:|---|
| `generation_dense_down_gemm` | `0.0104925866666667` | `sol` |
| `generation_dense_ffn_ar` | `0.000191146666666667` | `sol` |
| `generation_dense_ffn_norm` | `4.096e-05` | `sol` |
| `generation_dense_gate_up_gemm` | `0.0209800533333333` | `sol` |
| `generation_dense_swiglu` | `1.024e-05` | `sol` |
| `generation_embedding` | `2.56e-06` | `sol` |
| `generation_embedding_ar` | `4.77866666666667e-05` | `sol` |
| `generation_full_mla_approx_attention` | `0.000454666666666667` | `sol` |
| `generation_full_mla_approx_attention_ar` | `0.000955733333333333` | `sol` |
| `generation_full_mla_approx_attn_norm` | `0.0002048` | `sol` |
| `generation_full_mla_approx_bmm_post` | `0.00441173333333333` | `sol` |
| `generation_full_mla_approx_bmm_pre` | `0.00441173333333333` | `sol` |
| `generation_full_mla_approx_downscale_gemm` | `0.0541016` | `sol` |
| `generation_full_mla_approx_proj_gemm` | `0.0524629333333333` | `sol` |
| `generation_full_mla_approx_q_b_proj_gemm` | `0.01968` | `sol` |
| `generation_logits_gemm` | `0.0412559933333333` | `sol` |
| `generation_moe_ffn_norm` | `0.00077824` | `sol` |
| `generation_moe_overlap` | `0.70279936` | `sol` |
| `generation_moe_shared_merge` | `0.00058368` | `sol` |
| `generation_p2p` | `0.00024576` | `sol` |
| `generation_swa_mla_approx_attention` | `0.001364` | `sol` |
| `generation_swa_mla_approx_attention_ar` | `0.0028672` | `sol` |
| `generation_swa_mla_approx_attn_norm` | `0.0006144` | `sol` |
| `generation_swa_mla_approx_bmm_post` | `0.0132352` | `sol` |
| `generation_swa_mla_approx_bmm_pre` | `0.0132352` | `sol` |
| `generation_swa_mla_approx_downscale_gemm` | `0.1623048` | `sol` |
| `generation_swa_mla_approx_proj_gemm` | `0.1573888` | `sol` |
| `generation_swa_mla_approx_q_b_proj_gemm` | `0.05904` | `sol` |

#### Aggregate `mix_step`

| Operation | Latency ms | Source |
|---|---:|---|
| `context_dense_down_gemm` | `0.0113595733333333` | `sol` |
| `context_dense_ffn_ar` | `0.0244667733333333` | `sol` |
| `context_dense_ffn_norm` | `0.00524288` | `sol` |
| `context_dense_gate_up_gemm` | `0.0220637866666667` | `sol` |
| `context_dense_swiglu` | `0.00131072` | `sol` |
| `context_embedding` | `0.00032768` | `sol` |
| `context_embedding_ar` | `0.00611669333333333` | `sol` |
| `context_full_mla_approx_attention (scaled)` | `0.008192` | `sol` |
| `context_full_mla_approx_attention_ar (scaled)` | `0.122333866666667` | `sol` |
| `context_full_mla_approx_attn_norm (scaled)` | `0.0262144` | `sol` |
| `context_full_mla_approx_downscale_gemm (scaled)` | `0.0584704` | `sol` |
| `context_full_mla_approx_kv_b_proj_gemm (scaled)` | `0.0111957333333333` | `sol` |
| `context_full_mla_approx_proj_gemm (scaled)` | `0.0567978666666667` | `sol` |
| `context_full_mla_approx_q_b_proj_gemm (scaled)` | `0.0221184` | `sol` |
| `context_logits_gemm` | `0.0412559933333333` | `sol` |
| `context_moe` | `38.46373376` | `sol` |
| `context_moe_ffn_norm` | `0.09961472` | `sol` |
| `context_moe_post_dispatch` | `0.464868693333333` | `sol` |
| `context_moe_pre_dispatch` | `0` | `sol` |
| `context_moe_router_gemm` | `0.126593706666667` | `sol` |
| `context_moe_shared_merge` | `0.07471104` | `sol` |
| `context_p2p` | `0.03145728` | `sol` |
| `context_shared_down_gemm` | `0.0378743466666667` | `sol` |
| `context_shared_ffn_ar` | `0.464868693333333` | `sol` |
| `context_shared_gate_up_gemm` | `0.0632968533333333` | `sol` |
| `context_shared_swiglu` | `0.00311296` | `sol` |
| `context_swa_mla_approx_attention (scaled)` | `0.024576` | `sol` |
| `context_swa_mla_approx_attention_ar (scaled)` | `0.3670016` | `sol` |
| `context_swa_mla_approx_attn_norm (scaled)` | `0.0786432` | `sol` |
| `context_swa_mla_approx_downscale_gemm (scaled)` | `0.1754112` | `sol` |
| `context_swa_mla_approx_kv_b_proj_gemm (scaled)` | `0.0335872` | `sol` |
| `context_swa_mla_approx_proj_gemm (scaled)` | `0.1703936` | `sol` |
| `context_swa_mla_approx_q_b_proj_gemm (scaled)` | `0.0663552` | `sol` |
| `generation_attention (not executed)` | `0` | `not_executed` |

#### Disaggregate `decode`

| Operation | Latency ms | Source |
|---|---:|---|
| `generation_dense_down_gemm` | `0.0113319936` | `sol` |
| `generation_dense_ffn_ar` | `0.0002064384` | `sol` |
| `generation_dense_ffn_norm` | `4.42368e-05` | `sol` |
| `generation_dense_gate_up_gemm` | `0.0226584576` | `sol` |
| `generation_dense_swiglu` | `1.10592e-05` | `sol` |
| `generation_embedding` | `2.7648e-06` | `sol` |
| `generation_embedding_ar` | `5.16096e-05` | `sol` |
| `generation_full_mla_approx_attention` | `0.000488448` | `sol` |
| `generation_full_mla_approx_attention_ar` | `0.001032192` | `sol` |
| `generation_full_mla_approx_attn_norm` | `0.000221184` | `sol` |
| `generation_full_mla_approx_bmm_post` | `0.004764672` | `sol` |
| `generation_full_mla_approx_bmm_pre` | `0.004764672` | `sol` |
| `generation_full_mla_approx_downscale_gemm` | `0.058429728` | `sol` |
| `generation_full_mla_approx_proj_gemm` | `0.056659968` | `sol` |
| `generation_full_mla_approx_q_b_proj_gemm` | `0.0212544` | `sol` |
| `generation_logits_gemm` | `0.0445564728` | `sol` |
| `generation_moe_ffn_norm` | `0.0008404992` | `sol` |
| `generation_moe_overlap` | `0.7590233088` | `sol` |
| `generation_moe_shared_merge` | `0.0006303744` | `sol` |
| `generation_p2p` | `0.0002654208` | `sol` |
| `generation_swa_mla_approx_attention` | `0.001465344` | `sol` |
| `generation_swa_mla_approx_attention_ar` | `0.003096576` | `sol` |
| `generation_swa_mla_approx_attn_norm` | `0.000663552` | `sol` |
| `generation_swa_mla_approx_bmm_post` | `0.014294016` | `sol` |
| `generation_swa_mla_approx_bmm_pre` | `0.014294016` | `sol` |
| `generation_swa_mla_approx_downscale_gemm` | `0.175289184` | `sol` |
| `generation_swa_mla_approx_proj_gemm` | `0.169979904` | `sol` |
| `generation_swa_mla_approx_q_b_proj_gemm` | `0.0637632` | `sol` |

#### Disaggregate `prefill`

| Operation | Latency ms | Source |
|---|---:|---|
| `context_dense_down_gemm` | `0.0124955306666667` | `sol` |
| `context_dense_ffn_ar` | `0.0269134506666667` | `sol` |
| `context_dense_ffn_norm` | `0.005767168` | `sol` |
| `context_dense_gate_up_gemm` | `0.0242701653333333` | `sol` |
| `context_dense_swiglu` | `0.001441792` | `sol` |
| `context_embedding` | `0.000360448` | `sol` |
| `context_embedding_ar` | `0.00672836266666667` | `sol` |
| `context_full_mla_approx_attention` | `0.0090112` | `sol` |
| `context_full_mla_approx_attention_ar` | `0.134567253333333` | `sol` |
| `context_full_mla_approx_attn_norm` | `0.02883584` | `sol` |
| `context_full_mla_approx_downscale_gemm` | `0.06431744` | `sol` |
| `context_full_mla_approx_kv_b_proj_gemm` | `0.0123153066666667` | `sol` |
| `context_full_mla_approx_proj_gemm` | `0.0624776533333333` | `sol` |
| `context_full_mla_approx_q_b_proj_gemm` | `0.02433024` | `sol` |
| `context_logits_gemm` | `0.0453815926666667` | `sol` |
| `context_moe` | `42.310107136` | `sol` |
| `context_moe_ffn_norm` | `0.109576192` | `sol` |
| `context_moe_post_dispatch` | `0.511355562666667` | `sol` |
| `context_moe_pre_dispatch` | `0` | `sol` |
| `context_moe_router_gemm` | `0.139253077333333` | `sol` |
| `context_moe_shared_merge` | `0.082182144` | `sol` |
| `context_p2p` | `0.034603008` | `sol` |
| `context_shared_down_gemm` | `0.0416617813333333` | `sol` |
| `context_shared_ffn_ar` | `0.511355562666667` | `sol` |
| `context_shared_gate_up_gemm` | `0.0696265386666667` | `sol` |
| `context_shared_swiglu` | `0.003424256` | `sol` |
| `context_swa_mla_approx_attention` | `0.0270336` | `sol` |
| `context_swa_mla_approx_attention_ar` | `0.40370176` | `sol` |
| `context_swa_mla_approx_attn_norm` | `0.08650752` | `sol` |
| `context_swa_mla_approx_downscale_gemm` | `0.19295232` | `sol` |
| `context_swa_mla_approx_kv_b_proj_gemm` | `0.03694592` | `sol` |
| `context_swa_mla_approx_proj_gemm` | `0.18743296` | `sol` |
| `context_swa_mla_approx_q_b_proj_gemm` | `0.07299072` | `sol` |

Every executed entry is `sol`. The single aggregate `generation_attention (not executed)` marker has `0 ms` and source `not_executed` by public API contract; it is not counted as executed work.

### CLI Estimate Evidence

| Metric | SDK Expected | CLI Actual | Absolute Delta | Result |
|---|---:|---:|---:|---|
| TTFT | `105.16356682000 ms` | `105.164 ms` | `0.00043318000 ms` | PASS after display rounding |
| TPOT | `1.32416116666667 ms` | `1.324 ms` | `0.000161166666666546 ms` | PASS after display rounding |
| Request latency | `TTFT + TPOT = 106.4877279867 ms` | `106.488 ms` | `0.0002720133 ms` | PASS after display rounding |
| tokens/s | Display-only | `18.78` | N/A | Recorded |
| tokens/s/gpu | Display-only | `1.17` | N/A | Recorded |
| tokens/s/user | Display-only | `755.20` | N/A | Recorded |
| seq/s | Display-only | `18.782` | N/A | Recorded |
| Concurrency | Expected `2` | `2` | `0` | PASS |
| Memory/GPU | Display-only | `91.31 GB` | N/A | Recorded |

CLI return code: `0`; stderr bytes: `0`.

### CLI Generate and Generic Naive-Sizing Evidence

| Metric | Authoritative / Expected | CLI Actual | Delta | Result / Interpretation |
|---|---:|---:|---:|---|
| Parameter count | `1,490,676,214,656` | `1,559,313,383,424` | `68,637,168,768` (`4.604431739983%`) | MISMATCH documented; generic estimator is non-authoritative |
| BF16 model size | `2776.600820302963 GiB` | `2904.447509765625 GiB` | `127.846689462662 GiB` | MISMATCH documented |
| Required TP | Must fit within available 8 for an eight-GPU feasibility claim | `32` | `+24` above maximum | `fit=False`; no feasibility claim |
| Maximum TP | `8` | `8` | `0` | Recorded |
| Rendered artifacts | `7` smoke artifacts | `7` | `0` | PASS as identity/rendering smoke only |
| Internal strategy | Not a validated recommendation | `TEP=8` | N/A | Recorded only |
| Display summary | Not a validated recommendation | `TP=1, PP=1` | N/A | Recorded only |

The root cause is `generator.naive._estimate_model_weight_bytes()`: it applies one embedding, one `4 * H^2` attention formula, and a 512-expert MoE FFN to all 80 layers. It does not consume `block_types`, the four dense layers, shared-expert geometry, both embeddings, or the CSV attention totals. MTP is not included and is not the cause. No generator source was modified.

Generated temporary artifacts:

| Relative Path | Bytes | SHA256 |
|---|---:|---|
| `stepfun-ai_Step4-Pro-V1_naive_tp1_pp1_731664/agg_config.yaml` | `904` | `f533a833718689bcb7f4d3a7e93818d61ed4c8a01001aa5275e995bfb7c04b63` |
| `stepfun-ai_Step4-Pro-V1_naive_tp1_pp1_731664/bench_run.sh` | `2001` | `bb9026f1311a8ba4f517e261710776e339db5b2bfe46dec249aca684ee1ae9a2` |
| `stepfun-ai_Step4-Pro-V1_naive_tp1_pp1_731664/generator_config.yaml` | `2208` | `9d054bd28821edf031464cd9fa2f0faa8ffa61e553c772d836361d80c8ead571` |
| `stepfun-ai_Step4-Pro-V1_naive_tp1_pp1_731664/k8s_bench.yaml` | `2100` | `6c3a0d728d99f54f554dbab06d72c1380e4d48dcd1f1865c4398a5c1a5c98a6e` |
| `stepfun-ai_Step4-Pro-V1_naive_tp1_pp1_731664/k8s_deploy.yaml` | `3878` | `cf4de13b1b139f9c7c78055aed42afde5e023596d181c3314a277e6b02ed998f` |
| `stepfun-ai_Step4-Pro-V1_naive_tp1_pp1_731664/run_0.sh` | `913` | `c337217083c6e6c7fe975fe13da6ee0ca98f18d31e2f02ff123bd0f4d6e75022` |
| `stepfun-ai_Step4-Pro-V1_naive_tp1_pp1_731664/sflow.yaml` | `11545` | `5df1d7f405b93c86f2556363e7224a44cb8df8ce93abb2308eb715882433d17c` |

The CLI warning was present verbatim: the configuration was generated without memory validation or performance optimization and may not work if the model is too large. Return code was `0`; this proves rendering, not fit.

### Environment and Validation Failures: Root Cause and Resolution

#### `/usr/bin/time`

- **Failure:** `/bin/bash: /usr/bin/time: No such file or directory`, exit `127`; pytest did not start.
- **Root cause:** The binary is not installed on this host.
- **Resolution:** Run pytest directly and record its built-in elapsed time and shell exit code. No dependency or source change was needed.

#### Long `TMPDIR` and AF_UNIX

- **Failure:** initial full unit `13 failed / 2050 passed / 12 skipped / 1123 deselected`, all failures in `test_parallel_run.py` with child-side `OSError: AF_UNIX path too long`.
- **Measured values:** worktree `PWD` length `70`; task-local `TMPDIR` length `81`; representative manager listener path length `113`.
- **Root cause:** Python SyncManager appends `pymp-*/listener-*` below `TMPDIR`; the result cannot fit Linux AF_UNIX `sun_path`.
- **Controlled proof:** identical single test failed with long `TMPDIR` and passed with `/tmp`; both complete collector runs passed using verified short paths.
- **Resolution:** final full unit used the preflighted short `/tmp`, passed with zero failures, and required no code/test change.

#### Ruff format scope

- **Diagnostic:** `ruff format --check .` reported four Python copies nested under `tests/.tmp/pytest-of-i-fengyicheng/...`.
- **Root cause:** pytest fixtures copied tracked performance scripts into an untracked evidence directory; the recursive formatter included those non-delivery copies.
- **Resolution:** preserve the temp evidence as required, verify all 432 Git-tracked Python files, rerun with `--exclude tests/.tmp`, and retain the original diagnostic in this report.

### Independent Final Review

| Item | Actual |
|---|---|
| Backend | StepCode Claude `claude-opus-4-6[1m]`, `effort=max` |
| Artifact | `.omx/artifacts/claude-act-as-the-independent-final-code-reviewer-for-the-step4-pro-2026-07-15T19-36-24-039Z.md` |
| Artifact bytes | `12,336` |
| Artifact SHA256 | `15669234b47250e4b9d1f07577f90b942139fbf4eeaa710b4283adad05aec9b2` |
| Verdict | `APPROVE` |
| Blocking findings | `0` |
| Delivery authorization | Reviewer explicitly approved final archival/reporting |

Reviewer observations: the shared CustomAllReduce tuple correction is semantically necessary; Step4-local routed-MoE assignments intentionally bypass generic fallback behavior; six positive-integer checks are valid; exact CLI warning text and private test-helper access are minor maintenance risks. No code remediation was required.

## 4. Acceptance Conclusion

All approved Step4-Pro-V1 support criteria pass. The cached model topology and CSV arithmetic are exact; full graph execution is formula-only in `SOL`; direct `SOL_FULL` roofline tuples close; aggregate/disaggregate SDK and offline CLI paths resolve the model; original Step4 and full-unit regressions pass; delivery files are lint/format/diff clean; and independent review returned `APPROVE`.

The following are intentionally **not** claimed as complete support: Task-level `SOL_FULL`, AFD, faithful Step4-Pro-V1 attention/KV-cache modeling, empirical profile coverage, silicon support-matrix certification, eight-GPU fit, or a block-aware generic naive estimator. These remain documented future items and were not hidden by fallback or calibration.

### Evidence Artifact Retention

- Numeric evidence JSON: `tests/.tmp/step4_pro_v1_numeric_evidence.json`, `32,593` bytes, SHA256 `ec8d9a1b37f343a56aa4ef52b57e1ebca2f33685eca9f0e4cc25a3ea39582555`.
- The JSON and generated CLI artifacts remain untracked under `tests/.tmp/` and are not part of the product commit. Their required numeric content is transcribed above.

## 5. Follow-up Roofline-Review Document Validation

### Test Script Information

- Document: `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/docs/step4_pro_v1_roofline_model_review.md`
- Focused test files:
  - `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/tests/unit/sdk/models/test_step4_pro_v1.py`
  - `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/tests/unit/sdk/database/test_step4_pro_v1_roofline.py`
  - `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/tests/integration/test_step4_pro_v1_support.py`
- Environment: conda env `/home/i-fengyicheng/miniconda3/envs/aic-step-design`, Python `3.11.15`, pytest `8.4.2`, `PYTHONPATH=src:.`, `MPLBACKEND=Agg`, `TMPDIR=/data/ycfeng/tmp`.
- Reproducible focused command:

```bash
cd /data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro
PYTHONPATH=src:. MPLBACKEND=Agg TMPDIR=/data/ycfeng/tmp \
  /home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python -m pytest -q \
  tests/unit/sdk/models/test_step4_pro_v1.py \
  tests/unit/sdk/database/test_step4_pro_v1_roofline.py \
  tests/integration/test_step4_pro_v1_support.py
```

### Validation Criteria

- One standalone document contains both required levels.
- Inventory count equals `33` context top-level ops and `28` generation top-level ops; generation includes `8` nested children, `36` recursive nodes, and `35` executable leaves.
- Every source `path:line-range` exists.
- Projection widths, TP8 local widths, Attention gaps, and KV mismatch reproduce exactly.
- Existing model/roofline/integration behavior remains unchanged.
- Independent Claude returns no BLOCK.

### Test Results and Numeric Evidence

| Check | Expected | Actual | Delta / Result |
|---|---:|---:|---|
| Context top-level operations | 33 | 33 | 0, PASS |
| Generation top-level operations | 28 | 28 | 0, PASS |
| Generation recursive / executable leaves | 36 / 35 | 36 / 35 | 0 / 0, PASS |
| Source references | all valid | 44 valid, 0 errors | PASS |
| Markdown table rows | complete two-level artifact | 241 | PASS |
| Projection widths | 2,112 / 24,576 / 32,768 / 16,384 | exact match | 0, PASS |
| TP8 local widths | 3,072 / 4,096 / 2,048 / 2,048 / 256 / 16,112 | exact match | 0, PASS |
| Full Attention gap | 39,849,024 / 26.0289125137% | exact match | 0, PASS |
| SWA Attention gap | 50,333,792 / 23.5301782164% | exact match | 0, PASS |
| Temporary KV vs CSV | 48.31838208 GB vs 10.7 GB | gap 37.61838208 GB; ratio 4.515736642991x | PASS audit |
| Focused regression | zero failures | 65 passed in 13.35 s | PASS, exit 0 |
| `git diff --check` | 0 errors | 0 errors | PASS, exit 0 |
| Independent review | APPROVE/WATCH, no BLOCK | APPROVE; 0 Critical, 0 Important, 3 Minor | PASS |

Markdown lint was not available on the host (`markdownlint` and `markdownlint-cli2` were absent). The document was instead checked for heading hierarchy, table presence, source references, formula arithmetic, whitespace, and focused behavioral regression.

Independent review artifact:

```text
.omx/artifacts/claude-act-as-the-independent-technical-reviewer-for-the-documentat-2026-07-16T07-37-30-818Z.md
bytes: 7,407
SHA256: b8cf0d82d4c1af74b260eebe54646d7dfbbba4fffd821c31f66b0cfdae93a1ab
verdict: APPROVE
```

Two Minor clarifications were applied: the numeric fixture is now explicitly labeled as simplified rather than real H200 peak, and the Full/SWA standard-GQA mismatch includes a complete derivation. The third line-range observation was checked against the current source; `moe.py:1333-1355` is the exact vLLM branch, so no change was appropriate.
