## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-15 | Recorded formatter-consistent corrected-v3 provenance, audit root causes, and minimum-cost delivery constraints |
| 2026-07-10 | Created operational notes for Step4 predefined ops planning |
| 2026-07-10 | Added accepted Step4 non-full attention decision |
| 2026-07-12 | Added Step4Air YAML/AELLO implementation evidence and source-of-truth boundary |
| 2026-07-12 | Recorded gated SwiGLU dense FFN decision and numeric reconciliation |
| 2026-07-12 | Recorded shared-expert FP8 projection and BF16 norm/merge decision |
| 2026-07-12 | Recorded missing Attention Detail source and numeric topology discrepancy |
| 2026-07-12 | Distinguished checkpoint nextn layers from runtime speculative proposals |
| 2026-07-12 | Recorded OMX question transport timeout and plain-text grilling continuation |
| 2026-07-13 | Recorded accepted checkpoint nextn=3 and normalized comparison policy |
| 2026-07-13 | Confirmed AELLO GQA has no hidden projection that reconciles CSV values |
| 2026-07-13 | Confirmed ignored local CSV has no recoverable Git history or detail sheet |
| 2026-07-13 | Recorded repeated OMX question timeout for attention source precedence |
| 2026-07-13 | Recorded temporary MLA substitution direction and its unresolved modeling scope |
| 2026-07-13 | Recorded all-92-layer MLA scope with retained Full/SWA labels |
| 2026-07-13 | Recorded independent WATCH review for DeepSeek-V3-style temporary MLA geometry |
| 2026-07-13 | Recorded accepted DeepSeek-V3 generic MLA geometry and granular SOL composition |
| 2026-07-13 | Recorded HuggingFace API timeout during Step4 naming verification |
| 2026-07-13 | Recorded OMX question timeout for local model identity decision |
| 2026-07-13 | Recorded accepted local Step4 model identity contract |
| 2026-07-13 | Validated the test-side simulated H800 system spec and recorded numeric values |
| 2026-07-13 | Recorded accepted test-side simulated H800 SOL source |
| 2026-07-13 | Recorded binary ISL mapping for the requested `iso` values |
| 2026-07-13 | Recorded primary `osl=1` matrix and separate `4K/1024` decode smoke |
| 2026-07-13 | Confirmed `step-design` as the intended implementation branch |
| 2026-07-13 | Recorded AIC prefix-cache field semantics before prefix-policy grilling |
| 2026-07-13 | Recorded explicit `prefix=0` workload policy |
| 2026-07-13 | Recorded required coverage of both `agg` and `disagg` serving modes |
| 2026-07-13 | Recorded exact TTFT-only SLA and ranking contract from the referenced Phase 2 task |
| 2026-07-13 | Recorded root cause of asymmetric chunked-prefill modeling in `agg` vs `disagg` |
| 2026-07-13 | Recorded aligned chunked-prefill-disabled validation policy |
| 2026-07-13 | Audited Step4 `352`-expert feasibility and AIC/vLLM uneven-EP semantics |
| 2026-07-13 | Recorded the accepted common 21-config parallel space and exhaustive batch-cap policy |
| 2026-07-13 | Recorded the verified conda environment for final runner-based docs validation |
| 2026-07-13 | Audited reference-search parity and recorded missing invariants, role fields, orchestration limits, topology, OOM, and neutral correction policy |
| 2026-07-13 | Recorded implementation authorization, isolated worktree, and clean-baseline environment requirements |
| 2026-07-13 | Recorded direct-SOL Step4 model-composition constraints from the completed explorer |
| 2026-07-13 | Aligned workload-specific ranking notes with the accepted primary Prefill-throughput decision |

# Notes: Step4 Predefined Ops Planning

## Stage Constraints

- The plan/discussion gate is complete, and the user explicitly authorized implementation.
- All implementation and validation run in `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-predefined-ops` on branch `task/step4-predefined-ops`.
- The original `/data/ycfeng/stepfun-performance-optimization/aiconfigurator` checkout remains untouched because it contains unrelated dirty changes.
- Subprocess-based tests must use the absolute `PYTHONPATH=/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-predefined-ops/src`; a relative `PYTHONPATH=src` is not sufficient for forked worker tests.
- The verified environment is `/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python` with Python `3.11.15`.
- No `rm` or `mv` operations are permitted without explicit user approval.

## Direct-SOL Model-Composition Handoff

- Step4 does not require a new operation class. Compose `Step4Model` from existing `GEMM`, `ElementWise`, `ContextMLA`, `GenerationMLA`, `MLABmm`, `MoE`, and `MoEDispatch` primitives.
- Use `GEMMQuantMode.fp8` for FP8 Step4 projections. Do not use `fp8_static`; its extra scale-estimation path changes provenance to `estimated` and violates the required all-SOL result contract.
- Direct-SOL validation must cover individual non-zero primitive results and their aggregate, with `PerformanceResult.source == "sol"` in both cases.
- Monkeypatch relevant data loaders to fail on call, recursively reject profiled/fallback module classes, and test both pre-dispatch and post-dispatch `MoEDispatch` on the target vLLM path; checking routed `MoE` alone is insufficient.
- The MTP grilling round's OMX question process returned no answer payload after approximately `330` seconds and was interrupted with `Ctrl-C`; continue that one question through the plain-text transport without changing its choices.
- The attention source-precedence round repeated the same transport failure on 2026-07-13: `omx question` returned no answer payload after approximately `600` seconds and was interrupted with `Ctrl-C`. Continue the unchanged single question through plain-text transport.
- The model-identity round also returned no answer payload after approximately `480` seconds and was interrupted with `Ctrl-C`. Preserve the unchanged naming choices through plain-text transport and infer no default.

## Repository State Observed

- Original working directory: `/data/ycfeng/stepfun-performance-optimization/aiconfigurator`
- Original Git branch observed by `git branch --show-current`: `step-design`
- Isolated implementation worktree: `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-predefined-ops`
- Isolated implementation branch: `task/step4-predefined-ops`
- Worktree base: `step-design` commit `9ce84ebbe3a0d7f785c91d055bdbdf4fdaabcbf1`
- User-requested branch text: `stepfun-design`
- Accepted branch policy: the later implementation targets the active `step-design` branch; the original `stepfun-design` text is treated as a typo.
- Repository lookup found no local or remote `stepfun-design` ref and found `step-design` as the only branch matching `*step*design*`.
- Pre-existing dirty / untracked paths observed before this task's docs were created:
  - `M src/aiconfigurator/sdk/sweep.py`
  - `M src/aiconfigurator/sdk/task_v2.py`
  - `M tests/unit/sdk/sweep/test_sweep.py`
  - `M tests/unit/sdk/task_v2/test_task_config.py`
  - `?? .omc/`
  - `?? task_memory/`
  - `?? tests/performance/`
  - `?? tests/unit/performance/`
- These existing code/test modifications were not touched, stashed, committed, or copied. The isolated worktree resolves the implementation-safety requirement without altering the original checkout.
- Two 30-second HuggingFace API requests timed out while checking the StepFun organization and known Step3.7 model metadata. Public Step4 naming therefore could not be verified in this session; naming must be treated as an explicit local contract rather than an observed public fact.
- Accepted local model identity: model ID `stepfun-ai/Step4`, architecture `Step4ForCausalLM`, ModelFamily `STEP4`. The later config filename should follow the existing local-cache convention as `src/aiconfigurator/model_configs/stepfun-ai--Step4_config.json`.

## Source Artifacts Inspected

- `README.md`
  - Confirms AIC CLI modes and `--database-mode` behavior.
  - `SOL` is the Speed-of-Light / roofline-only mode described by the CLI docs.
- `AGENTS.md`
  - Confirms generator and collector guardrails; this task should not modify either area in the planning stage.
- `/data/ycfeng/stepfun-performance-optimization/permormancebenchmark/architecture_calculator_v1 - Main.csv`
  - Main source for Step4 and DS-V4-Pro architectural numbers.
- `src/aiconfigurator/model_configs/deepseek-ai--DeepSeek-V4-Pro_config.json`
  - Existing predefined DS-V4-Pro HF-style config.
- `src/aiconfigurator/model_configs/sgl-project--DeepSeek-V4-Pro-FP8_config.json`
  - Existing DS-V4-Pro variant config.
- `src/aiconfigurator/sdk/models/deepseek_v4.py`
  - Existing DS-V4-Pro predefined model ops pipeline.
- `src/aiconfigurator/sdk/operations/dsv4.py`
  - DS-V4-specific mHC / compressed attention / MegaMoE operations. Important risk: MegaMoE path is measured-data based and should not be reused for Step4 roofline-only modeling.
- `src/aiconfigurator/sdk/models/moe.py`
  - Existing generic MoE model pipeline using GQA attention + routed MoE operations.
- `src/aiconfigurator/sdk/models/qwen35.py`
  - Existing hybrid layer-type pattern for splitting ops by attention layer kind.
- `src/aiconfigurator/sdk/operations/attention.py`
  - Generic `ContextAttention` / `GenerationAttention` support `window_size` and `DatabaseMode.SOL` queries.
- `task_memory/task_2026-07-04_aic_roofline_pareto_search/result-vllm-only-latest/todo_search_configs_deepseek-v4-pro.md`
  - SHA256: `739fb928c956eb788d945d8610ad7cd0ffca7b2d578304d825fd3173ec8c8fb7`.
  - Defines TTFT SLA values `200`, `500`, `1000`, `2000`, and `5000` ms, with `tpot=50000` ms as an effective non-constraint.
- `task_memory/task_2026-07-04_aic_roofline_pareto_search/result-vllm-only-latest/dsv4pro_approved_run/summary.md`
  - SHA256: `1333d88c22650c88bd7594ef642b306ee56539aaa14aa6e50dbf7681471aa1e8`.
  - Confirms final best-config ranking by `tokens/s/gpu_cluster` and reports `240/240` aggregate/disaggregate status rows for the one-model reference run.
- `task_memory/task_2026-07-04_aic_roofline_pareto_search/test_report_2026-07-08_chunked_prefill_tp8_ep8_bs1_rca.md`
  - SHA256: `94c2e25637df92ea14a6a7f1a7bb223a117a21a8ce6cf0d564685ab4d6fde173`.
  - Establishes that current AIC truly applies chunked-prefill modeling only in `agg`; the `disagg` flag is serialized but not consumed.
- HuggingFace Step3.7 config URL inspected via official model raw config:
  - `https://huggingface.co/stepfun-ai/Step-3.7-Flash-NVFP4/raw/main/config.json`
- `/data/ycfeng/stepfun-performance-optimization/permormancebenchmark/aello/examples/exp_suite/models/step4air_prefill.yaml`
  - SHA256: `e4b361f73d33e1090214dd350e426ef7a59aac0f93f181bc0ce34019f7070cd1`
- `/data/ycfeng/stepfun-performance-optimization/permormancebenchmark/aello/examples/exp_suite/models/step4air_decode.yaml`
  - SHA256: `4f475222215b2bd6f1a5a7be06526e34aa3cc2f43f627a9bddc11d2bfec11e96`
- AELLO implementation paths used to interpret the YAML:
  - `aello/src/aello/models/components/builtin.py`
  - `aello/src/aello/models/deepseek/builder.py`
  - `aello/src/aello/modules/moe/configs.py`
  - `aello/src/aello/modules/moe/experts.py`
  - `aello/src/aello/modules/moe/moe.py`
  - `aello/src/aello/modules/tensor_parallel.py`
  - `aello/src/aello/precision/spec.py`
  - `aello/src/aello/precision/utils.py`

## Step3.7 Reference Facts

Use these only as fallback references when Step4 is ambiguous.

- Top-level architecture: `Step3p7ForConditionalGeneration`
- Text model architecture: `Step3p5ForCausalLM`
- Text hidden size: `4096`
- Text trunk layers: `45`
- Text layer types: `48` entries in the raw config; the first 45 correspond to the trunk, and the extra 3 align with nextn/MTP layers.
- Text attention pattern: every 4th layer is `full_attention`, the other layers are `sliding_attention`; trunk count is `12 full + 33 sliding`.
- Sliding window: `512`
- Dense/MoE split: `moe_layers_enum` begins at layer `3`, implying first 3 trunk layers are dense and layers 3-44 are MoE.
- MoE fields: `moe_num_experts=288`, `moe_top_k=8`, `moe_intermediate_size=1280`, `share_expert_dim=1280`.
- Nextn/MTP: `num_nextn_predict_layers=3`.
- Attention head fields contain a potential internal nuance: top-level text fields include `num_attention_heads=64`, `head_dim=128`, `num_attention_groups=8`, while `attention_other_setting` includes a separate `num_attention_heads=96`. Do not copy this blindly for Step4 without confirmation.

## MTP / nextn Semantics

- The Step3.7 checkpoint config declares `num_nextn_predict_layers=3`.
- Step4Air decode enables runtime speculative decoding with `proposals_per_iter=3`, `mtp.max_proposals=3`, and a single configured `acceptance_rate=0.80`.
- These are related but not interchangeable facts: checkpoint `num_nextn_predict_layers` describes available MTP layers, while `proposals_per_iter` describes how many proposals a runtime attempts per iteration.
- AIC resolves `Task.nextn` from the checkpoint's `num_nextn_predict_layers` when the user does not override it. A user override wins and emits a warning when it differs from the checkpoint.
- Existing AIC MoE families apply `nextn` to generation op counts through `1 / (1 + E[accepted extra tokens]) * (num_layers + nextn) / num_layers`; therefore selecting `nextn=3` changes modeled TPOT rather than merely adding config metadata.
- The AELLO scalar `acceptance_rate=0.80` cannot be copied directly into AIC's per-depth `nextn_accept_rates` list without a separate decision.
- Accepted policy: the Step4 predefined config declares `num_nextn_predict_layers=3`; the primary Step4-vs-DS-V4-Pro matrix explicitly overrides both models to `nextn=0`; Step4 `nextn=3` receives separate structural/smoke validation.
- The separate `nextn=3` check does not establish a comparative TPOT claim from an unconfirmed acceptance-rate mapping.

## Step4 Parameters Extracted from CSV

| Parameter | Step4 Value | Notes |
|---|---:|---|
| hidden size H | 4096 | CSV `model dim` |
| trunk layers | 92 | CSV says trunk only; MTP/nextn excluded |
| full attention layers | 23 | trunk full attention count |
| non-full attention layers | 69 | CSV labels as `SWA/linear/compressed/sparse` |
| dense FFN layers | 4 | placement not explicit |
| dense intermediate size | 13824 | dense SwiGLU inter size |
| routed experts | 352 | `num routed` |
| activated experts / top-k | 8 | `num activated` |
| latent MoE size | 0 | normal MoE formula, no latent MoE |
| per-routed expert inter size | 1536 | `moe_inter_size` |
| shared expert inter size | 1536 | `moe_shared_size` |
| vocab size | 128896 | same as Step3p5 |
| HC.enable | 0 | no DeepSeek-V4-style hyper-connection |
| HC.mult | 0 | no mHC multiplier |
| derived MoE layers | 88 | `92 - 4` |
| attention recipe | 23 full + 69 nonfull | CSV summary |
| param per full attention | 107,216,896 | CSV summary |
| param per nonfull attention avg | 107,216,896 | same as full in CSV |
| attention total params | 9,863,954,432 | CSV summary |
| total params with embedding | 598,040,346,624 | CSV summary |
| total activation | 25,619,562,496 | CSV summary |
| sparsity | 4.29% | CSV summary |

## DS-V4-Pro Comparison Parameters Extracted

| Parameter | DS-V4-Pro Value | Source |
|---|---:|---|
| hidden size H | 7168 | CSV + local config |
| trunk layers | 61 | CSV + local config |
| full attention layers | 0 | CSV |
| non-full attention layers | 61 | CSV |
| dense FFN layers | 0 | CSV |
| routed experts | 384 | CSV + local config |
| activated experts / top-k | 6 | CSV + local config |
| per-routed expert inter size | 3072 | CSV + local config |
| shared expert inter size | 3072 | CSV + local config `n_shared_experts=1` |
| vocab size | 129280 | CSV |
| HC.enable | 1 | CSV |
| HC.mult | 4 | CSV |
| model architecture | `DeepseekV4ForCausalLM` | local config |
| max position embeddings | 1048576 | local config |
| nextn/MTP | 1 | local config `num_nextn_predict_layers=1` |
| total params with embedding | 1,572,997,057,630 | CSV |

## Accepted Modeling Decisions

- Step4 `69 non-full attention layers` are GQA sliding-window attention with `96` query heads, `8` KV heads, qk/v head dimension `128`, and `window_size=512`.
- Step4 `23 full attention layers` are full GQA with `64` query heads, `8` KV heads, and qk/v head dimension `128`.
- Full and sliding attention therefore require distinct head topology; they cannot be represented as one shared `64/8` or `96/8` topology with only the window changed.
- Attention projection/output and KV-cache append/read are declared FP8 in both YAML files.
- Dense FFN layers form a leading prefix. Mapping the YAML ordering rule onto the CSV count gives first `4` trunk layers dense and remaining `88` trunk layers MoE.
- Each MoE layer includes the CSV-mandated gated shared expert with intermediate size `1536`; its gate/up/down projections are FP8, its norm is BF16, and its output is merged with the routed branch in BF16.
- Implementation should not use DS-V4 compressed/sparse/mHC-specific attention ops for Step4's non-full layers unless a later user-provided Step4 config explicitly changes this requirement.

## YAML-vs-CSV Source-of-Truth Boundary

| Concern | Authoritative source | Step4 value/use |
|---|---|---|
| Trunk layers, full/SWA counts, dense/MoE counts | Architecture CSV | `92`; `23/69`; `4/88` |
| Hidden/intermediate dimensions and expert topology | Architecture CSV | H=`4096`; dense=`13824`; routed/shared=`1536`; experts=`352`; top-k=`8` |
| Full/SWA attention algorithm and head topology | Step4Air YAML | full GQA=`64/8`; SWA GQA=`96/8`, window=`512`; head dims=`128` |
| Attention and KV-cache quantization | Step4Air YAML | FP8 projection/output and FP8 KV-cache read/append |
| Dense/MoE FFN graph and phase precision intent | Step4Air YAML plus AELLO component code | See sections below |
| TP/EP, gather/reduce-scatter, `shard_seq` | YAML example only | Do not hard-code into Step4 architecture |
| YAML layer counts | YAML example only | Ignore `3 + 11 + 31`; use CSV counts |

## Step4Air Attention Evidence

| Block | Attention | Query/KV heads | qk/v dim | Window | Precision intent |
|---|---|---:|---:|---:|---|
| `dense_swa_block` | GQA SWA | `96/8` | `128/128` | `512` | q/kv/output FP8; KV-cache append/read FP8 |
| `moe_full_block` | full GQA | `64/8` | `128/128` | none | q/kv/output FP8; KV-cache append/read FP8 |
| `moe_swa_block` | GQA SWA | `96/8` | `128/128` | `512` | q/kv/output FP8; KV-cache append/read FP8 |

- Prefill sets `shard_seq: true`; decode does not. This is a concrete phase/sharding choice, not a model-architecture constant.
- The YAML's `DeepseekV3Model` class is an AELLO builder wrapper. It does not imply Step4 uses DeepSeek MLA, compressed attention, mHC, or DS-V4 measured ops.

## Attention Parameter Source Discrepancy

- The Main CSV says its attention summary is assembled from or weighted by an `Attention Detail` sheet, but a workspace-wide filename/content search found no corresponding sheet, CSV export, workbook, or formula source.
- The Main CSV is explicitly ignored by `permormancebenchmark/.gitignore`, is not tracked on any local/remote branch, and has no Git history. Repository history also contains no `Attention Detail`, `.xlsx`, or `.ods` artifact from which the formulas can be recovered.
- Observed Main CSV SHA256 is `7639702f77ec0c387e06f5064f129608f498d49ab9e555ab1898fb18fcf38e89`; observed mtime is `2026-07-03 12:29:55 +0800`.
- Using the explicit YAML GQA topology and standard Q/K/V/O projection matrices gives:

| Branch | CSV expected params/layer | Q/K/V/O computed params/layer | Computed - expected | Relative delta |
|---|---:|---:|---:|---:|
| Full GQA `64/8` | 107,216,896 | 75,497,472 | -31,719,424 | -29.584% |
| SWA GQA `96/8` | 107,216,896 | 109,051,904 | +1,835,008 | +1.711% |

- Full computation: Q=`33,554,432`, K=`4,194,304`, V=`4,194,304`, O=`33,554,432`.
- SWA computation: Q=`50,331,648`, K=`4,194,304`, V=`4,194,304`, O=`50,331,648`.
- Static AELLO code inspection confirms these exact projection shapes: Q=`H x (num_heads * qk_head_dim)`, K=`H x (num_kv_heads * qk_head_dim)`, V=`H x (num_kv_heads * v_head_dim)`, and O=`(num_heads * v_head_dim) x H`, all without bias.
- The AELLO GQA component adds a `LayerNorm(H)` but no attention gate, low-rank branch, or other weighted projection. The Main CSV explicitly accounts for RMS/norm separately from `attention total param`, so that norm cannot be used to reconcile the attention summary.
- This discrepancy is a missing-source/provenance problem, not evidence that either branch should be multiplied by a calibration factor. The original `Attention Detail` data or an explicit source-precedence decision is required before exact parameter reconciliation can pass.

## Temporary MLA Substitution Direction

- User direction: temporarily use MLA as the Step4 attention substitute and explicitly declare the substitution.
- This is a declared modeling approximation, not a correction to the Step4 architecture evidence. The Step4Air YAML target remains Full GQA `64/8` plus SWA GQA `96/8, window=512`.
- The existing regular `ContextMLA` and `GenerationMLA` operations have direct `DatabaseMode.SOL` branches, so a roofline-only MLA estimate can avoid loading profiling-based perfdb data.
- Their current SOL equations are DeepSeek-shaped constants rather than config-driven Step4 geometry:
  - context uses a fixed per-head `(192 + 128)` attention width;
  - generation uses fixed `1088` operation width and `576` latent-cache width;
  - neither regular MLA op accepts `window_size`.
- DS-V4-specific module paths and `DeepSeekV4MegaMoEModule` must not be inferred from the word MLA. MegaMoE remains measured-data only and is prohibited for Step4's roofline-only requirement.
- Accepted scope: all `92` Step4 attention layers use the temporary MLA SOL latency substitute. Keep separate `23 Full` and `69 SWA` names/counts in the op inventory and reports for architecture traceability, but both groups use the same temporary MLA latency abstraction.
- Accepted temporary geometry: `num_heads=128`, `q_lora_rank=1536`, `kv_lora_rank=512`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, and `v_head_dim=128`, borrowed from the DeepSeek-V3 generic MLA baseline. Step4 keeps `hidden_size=4096` and its own layer/FFN/MoE parameters.
- Auditable projection dimensions for each temporary MLA layer:
  - fused latent projection: `4096 -> 2112`, where `2112 = 1536 + 512 + 64`;
  - query expansion: `1536 -> 24576`, where `24576 = 128 * (128 + 64)`;
  - KV expansion: `512 -> 32768`, where `32768 = 128 * (128 + 128)`;
  - output projection: `16384 -> 4096`, where `16384 = 128 * 128`.
- The planned SOL path is explicit granular projection GEMMs plus context/generation MLA core operations, including generation pre/post MLA BMM where required. Do not use `FallbackOp`, profiled `MLAModule`, or DS-V4-specific modules.
- Every later matrix row, chart, comparison, and test report must label the attention result as a temporary MLA approximation and must not claim faithful Full/SWA Step4 attention or CSV total-parameter reconciliation.
- Independent geometry review artifact: `.omx/artifacts/claude-you-are-an-independent-architecture-modeling-reviewer-review-2026-07-12T17-35-46-941Z.md`.
- Review verdict for the now-accepted DeepSeek-V3-style geometry is `WATCH`, conditional on granular projection GEMMs, direct SOL-only MLA core, retained `23/69` labels, attention/non-attention breakdown, and explicit approximation-dominated reporting at long ISL.
- The full-context-vs-window sequence-length term ratio is approximately `ISL / 512`: `8x`, `32x`, `128x`, `512x`, and `2048x` at `4k`, `16k`, `64k`, `256k`, and `1m`. This is a linear ratio of the attention sequence-length term, not a measured vLLM error or a guaranteed total-latency ratio.

## Step4Air FFN Evidence

### Dense FFN

- Declared precision: norm BF16; up/down projection FP8.
- YAML omits `gated`; `DenseFFNComponent` defaults `gated=False`.
- The realized graph is `norm -> up projection -> SiLU -> down projection`.
- This conflicts with the CSV formula, which explicitly counts three `H x I` projections. The user resolved the conflict in favor of gated SwiGLU.
- Final Step4 dense graph: combined gate/up projection -> split -> `SiLU(gate) * up` -> down projection.
- Numeric reconciliation:
  - `4096 * 13824 * 3 = 169,869,312` projection parameters per dense layer.
  - Four dense layers contribute `679,477,248` projection parameters.
  - This exactly matches the CSV `param per dense` formula/value.
- AELLO currently rejects tensor-parallel gated dense FFN, so copying the YAML's `tp_size=8` together with `gated=True` would fail fast.
- That AELLO simulator limitation does not redefine Step4 architecture and must not be copied into AIC's roofline graph.

### MoE FFN

- Routed MoE defaults `gated=True`; each routed expert is `gate/up projection -> SiLU(gate) * up -> down projection`.
- Dataflow: `norm -> routing gate logits -> GroupedTopK -> dispatch -> routed expert FFN -> combine`.
- Declared phase precision: norm/router/combine BF16; dispatch/experts and up/down FP8.
- `scoring_func` is not specified, so current AELLO default is softmax; `norm_topk_prob` defaults true; `imbalance_factor=1.0`.
- YAML does not provide `shared_intermediate_size`; current AELLO therefore disables the shared expert branch.
- CSV nevertheless requires a Step4 shared expert of size `1536`. The user resolved the YAML omission by selecting a gated shared branch with FP8 gate/up/down projections, BF16 norm, and BF16 merge with the routed output.
- This precision choice is an explicit user decision, not a value inferred from the two Step4Air YAML files.

## AELLO Realization Gaps to Avoid Copying Blindly

- YAML describes intended FP8 phases, but current AELLO plumbing does not necessarily turn every declaration into FP8 weight-memory metadata:
  - tensor-parallel dense FFN applies precision differently from the non-TP dense path;
  - MoE `quant_mode` is absent, so routed expert weight metadata can remain at the component default even when `ffn.experts.mode=fp8` is declared.
- These are simulator-realization details, not Step4 architecture facts. The AIC plan should encode the YAML's declared precision intent explicitly and test it, rather than reproduce an accidental metadata gap.

## Roofline-Only Design Notes

- AIC already exposes `--database-mode SOL` in CLI flows.
- Step4 should not reuse DS-V4 `DeepSeekV4MegaMoEModule`, because that module is explicitly tied to measured DS-V4 MegaMoE perf rows and rejects unsupported database modes.
- Step4 should prefer generic operations with SOL paths:
  - `GEMM`
  - `ElementWise`
  - `Embedding`
  - `ContextAttention` / `GenerationAttention` with explicit `window_size`
  - `MoEDispatch`
  - `MoE`
  - `CustomAllReduce`
  - `P2P`
- Verification should force `database_mode=SOL` and inspect outputs/artifacts to confirm Step4 latency does not rely on profiling-based perf data.

## Validation Environment Notes

- Built-in vLLM system data directories exist for `gb300`, `h200_sxm`, and `h100_sxm`.
- `h800_sxm` is not currently observed as a built-in system directory under `src/aiconfigurator/systems/data`; an existing custom test-side system file exists at `tests/performance/aic_roofline_pareto/systems/h800_sxm.yaml`.
- User's hardware names `h200`, `h100`, `h800` likely map to AIC system names `h200_sxm`, `h100_sxm`, `h800_sxm`; this should be stated clearly in the test plan and confirmed if needed.
- H800 custom YAML SHA256: `74b109e2cc30e3000b3c47707f6147d072361218873e3b60e4e12c795699dd26`.
- Loader smoke check succeeded with `--systems-paths default,tests/performance/aic_roofline_pareto/systems` in conda env `aic-step-design`:
  - simulation status=`simulated`;
  - memory capacity=`85,899,345,920` bytes (`80 GiB`);
  - memory bandwidth=`3,350,000,000,000` bytes/s;
  - dense BF16 Tensor Core=`989,500,000,000,000` FLOP/s;
  - FP8 Tensor Core=`1,979,000,000,000,000` FLOP/s;
  - GPUs/node=`8`.
- The custom spec is sufficient as a system-description source for direct SOL equations; it does not provide or imply H800 profiling perfdb data.
- Accepted H800 policy: use this test-side spec through `--systems-paths default,tests/performance/aic_roofline_pareto/systems`, keep it outside built-in package support, and mark every H800 result as `simulated` and `SOL-only`.
- Accepted sequence-length mapping: user-written `iso` means AIC `isl`; use exact binary values `4096`, `16384`, `65536`, `262144`, and `1048576` for `4k`, `16k`, `64k`, `256k`, and `1m` respectively.
- Accepted OSL policy: the five-point Step4-vs-DS-V4-Pro primary matrix uses explicit `osl=1` at every ISL, while a separate representative decode smoke uses `isl=4096, osl=1024`.
- The primary matrix and the independent decode smoke both use explicit `nextn=0`; the decode smoke covers ordinary generation ops and must not be conflated with the separate Step4-only `nextn=3` structural/smoke check.
- This split isolates long-context/prefill behavior from sustained decode behavior and avoids compounding the accepted full-context MLA approximation with long decode at the `1M` ISL boundary.
- AIC defines `prefix` as the number of tokens within `isl` whose KV is already cached. The CLI/API/Task default is `0`, but the validation plan must set it explicitly rather than inherit a default.
- For this task, `prefix=0` would force the full requested ISL through context/prefill modeling; a nonzero prefix would instead test prefix-cache reuse and change the amount of context work.
- Accepted prefix policy: set `prefix=0` explicitly in all five primary-matrix rows and in the independent `isl=4096, osl=1024` decode smoke. No additional prefix-cache reuse smoke is required by this task.
- Accepted serving-mode policy: run both `agg` and `disagg` for Step4 and DS-V4-Pro. The `disagg` runs must exercise separate prefill/decode configuration paths and allow worker counts above `8` under the total-GPU policy; `agg` supplies the same-model aggregate baseline.
- Accepted SLA/ranking policy from the referenced DeepSeek-V4-Pro Phase 2 task:
  - TTFT is the only active SLA dimension: `ttft=[200,500,1000,2000,5000]` ms.
  - TPOT is explicitly non-binding via `tpot=50000` ms and `pareto_sweep=false`; no TPOT-feasibility claim should be made.
  - Within each aligned model/system/workload/TTFT/serving-mode group, rank the five primary `OSL=1` workloads by descending cluster-normalized input/prefill token throughput. Rank the independent `4K/1024` decode smoke by descending output `tokens/s/gpu_cluster`. Still report TTFT, TPOT, request latency, and output/per-user throughput as observed metrics.
  - With `2` models, `4` systems, `6` workloads (`5` long-ISL points plus `1` decode smoke), `5` TTFT targets, and `2` serving modes, the planned comparison has `240` model-system-workload-TTFT points and `480` mode-run rows before separate Step4 `nextn=3` smoke coverage.
- Final runner-based docs validation must use `/home/i-fengyicheng/miniconda3/envs/aic-step-design/bin/python` (Python `3.11.15`, pandas `3.0.3`, AIC `0.10.0`). The repo `.venv` lacked pandas, while system Python lacked installed AIC distribution metadata; neither is a valid environment for importing `run_phase2_exp.py`.

## Chunked-Prefill Modeling Boundary

- `Task.sweep_agg_kwargs()` passes `enable_chunked_prefill` into `sweep_agg()`, which changes the context-token search through `_agg_ctx_tokens_list()`.
- `Task` exposes `prefill_enable_chunked_prefill` for `disagg`, but `Task.sweep_disagg_kwargs()` does not forward it and `sweep_disagg()` has no chunked-prefill parameter.
- The prior RCA found identical baseline/chunked disagg metrics for `22/22` common rows, confirming that the serialized flag did not change disagg modeling.
- Root cause: the `disagg` execution path has no data-flow edge from `prefill_enable_chunked_prefill` to prefill worker candidate generation; this is not a model-config or YAML problem.
- Enabling chunked prefill in both YAML modes would therefore create a hidden semantic mismatch: truly modeled in `agg`, label-only in `disagg`.
- This task must either disable chunked prefill in both modes, explicitly accept asymmetric coverage, or separately authorize a TDD plumbing change. No fallback or label-only claim is acceptable.
- Accepted policy: set aggregate `enable_chunked_prefill=false` and disaggregate `prefill_enable_chunked_prefill=false` explicitly in every required workload. Do not run an aggregate-only supplement in this task and do not modify the disaggregate plumbing as part of Step4 predefined-ops implementation.
- The missing disaggregate plumbing remains a documented future extension rather than a hidden fallback or an implied feature claim.

## Parallel Search-Space Feasibility Audit

- The referenced DeepSeek-V4-Pro Phase 2 search has `25` AIC-materialized configurations:
  - Pattern A pure EP: `21` configurations, including four `moe_ep=64` rows.
  - Pattern B pure MoE-TP: `4` configurations.
- Restricting Pattern A to `moe_ep<=32` leaves `17` pure-EP rows; with Pattern B, the aligned common search contains `21` configurations.
- DeepSeek-V4-Pro has `384` routed experts, so `384/64=6` and its four `EP=64` rows are evenly divisible.
- Step4 has `352` routed experts:
  - `352/32=11`, so `EP=32` is exact.
  - `352/64=5.5`, so `EP=64` requires uneven placement: `32` ranks with `6` experts and `32` ranks with `5` experts.
- AIC's `enumerate_parallel_config()` checks parallel-width equality but does not check `num_experts % moe_ep`; it therefore admits Step4 `EP=64` syntactically.
- The current generic AIC `MoE` SOL formula uses `num_experts // moe_ep_size` in resident-weight traffic. At `352/64`, it models `5` experts on every rank, effectively accounting for `320/352` experts in that term and omitting the heavier `6`-expert ranks:
  - global-equivalent omission=`32` experts (`9.091%` of routed experts);
  - average resident-expert undercount=`0.5/5.5=9.091%`;
  - heavy-rank resident-expert undercount=`1/6=16.667%`.
- Current upstream vLLM `main` at commit `36484e464a6cf763c5b4c8af7be8e19df324997a` supports uneven expert mapping when EPLB is disabled: `determine_expert_map()` assigns `base=floor(E/EP)` plus one expert to the first `E % EP` ranks. EPLB still rejects non-divisible expert counts, and kernel/all-to-all support varies by backend.
- Therefore, upstream runtime capability does not make the current AIC Step4 SOL estimate accurate at `EP=64`. The main comparison must either stop at the exact common intersection `EP<=32`, keep DS-only `EP=64` outside the aligned ranking, or explicitly expand scope to fix and test generic uneven-EP SOL modeling.
- Batch-policy facts from the reference runner:
  - sweep starts at batch size `1` with `batch_sweep_step=1`;
  - initial caps are agg=`1024`, prefill=`16`, decode=`1024`;
  - cap saturation is detected when the rank-1 batch equals a cap;
  - affected caps are doubled for reruns, and a still-saturated bounded run is marked non-final rather than accepted.

### Accepted final policy: `common_21_ep32`

- Step4 and DeepSeek-V4-Pro use the same exact `21` AIC-materialized parallel configurations in aggregate, disaggregate-prefill, and disaggregate-decode searches.
- Pattern A is the pure-EP candidate bundle below. After AIC validity filtering it materializes `17` configurations and covers `moe_ep=2,4,8,16,32`:

```yaml
num_gpu_candidates: [2, 4, 8, 16, 32]
tp_candidates: [1, 2, 4, 8]
pp_candidates: [1]
dp_candidates: [1, 2, 4, 8, 16, 32]
moe_tp_candidates: [1]
moe_ep_candidates: [2, 4, 8, 16, 32]
cp_candidates: [1]
```

- Pattern B is the pure-MoE-TP bundle below. It materializes `4` configurations with `tp=moe_tp=1,2,4,8`:

```yaml
num_gpu_candidates: [1, 2, 4, 8]
tp_candidates: [1, 2, 4, 8]
pp_candidates: [1]
dp_candidates: [1]
moe_tp_candidates: [1, 2, 4, 8]
moe_ep_candidates: [1]
cp_candidates: [1]
```

- Exact aligned count: `17 pure-EP + 4 pure-MoE-TP = 21` configurations.
- Fixed bounds: `pp=1`, `cp=1`, `tp<=8`; maximum modeled EP size=`32`; maximum worker size=`32`. This retains required `EP>8` and workers `>8` coverage through `16` and `32`.
- Explicit exclusions: no Step4 `EP=64`, no DeepSeek-V4-Pro `EP=64`, no DS-only supplement, no model-specific primary search space, and no generic uneven-EP SOL change in this task.
- Disaggregate mode retains all `AA`, `AB`, `BA`, and `BB` Pattern A/B prefill/decode bundle pairings.
- Batch search starts at `1` with step `1`. Initial caps are aggregate=`1024`, disaggregate prefill=`16`, and disaggregate decode=`1024`.
- If the rank-1 result equals an active cap, only that saturated cap is doubled and the affected search is rerun. This repeats as many times as needed; it is not limited to one rerun.
- Reruns stop only when the rank-1 batch is below the cap, the configuration is OOM, or the TTFT SLA is infeasible. Any still cap-saturated row is non-final and cannot enter final ranking.

## DeepSeek-V4-Pro Reference Search-Space Parity Audit

### Audit conclusion

- The existing `17+4=21` count was correct and remains the accepted common Step4/DeepSeek-V4-Pro space.
- The plan was under-specified rather than numerically wrong: it omitted complete invariants, the explicit 21-row table, role-prefixed Task fields, Pattern A/B experiment separation, disaggregate worker/replica ceilings, OOM adjudication, system-specific topology semantics, and effective correction/scaling values.
- The root cause was copying candidate bundles and aggregate counts without carrying forward the full deployment contract from the reference TODO, its generated YAMLs, current Task fields, and `enumerate_parallel_config()` invariants.

### Reference differences: omission versus deliberate override

| Dimension | DeepSeek-V4-Pro reference | Current Step4 comparison | Classification |
|---|---|---|---|
| Parallel rows | `21` pure EP through `EP=64` + `4` pure MoE-TP = `25` | `17` pure EP through `EP=32` + `4` pure MoE-TP = `21` | Deliberate user decision `common_21_ep32`; not an omission |
| `isl=524288` | Included | Excluded | Deliberate later user workload decision; five long-ISL points remain |
| OSL | `1024` for every reference ISL | `1` for five long-ISL points plus independent `4K/1024` decode smoke | Deliberate later user decision |
| Blackwell system | `b300_sxm` | `gb300` | Deliberate user request; these are distinct canonical systems, not aliases |
| Serving modes | `agg` and `disagg` | `agg` and `disagg` | Preserved |
| SLA | TTFT only; TPOT `50000` non-binding | Same | Preserved |
| Total GPUs/backend | `64`, vLLM | Same | Preserved |
| Batch start/step | `1/1` | Same, with later cap-expansion policy | Preserved and made stricter |

### Materialized search facts

- Core equalities: `worker_gpus=tp*dp*pp*cp=tp*dp` and `dp*tp*cp=moe_tp*moe_ep` because `pp=cp=1`.
- Pattern A: `moe_tp=1`, `moe_ep=tp*dp`, `tp*dp>=2`; count=`17`.
- Pattern B: `dp=1`, `moe_tp=tp`, `moe_ep=1`; count=`4`.
- Union count=`21`; `EP>8` count=`8`; `EP=16` count=`4`; `EP=32` count=`4`; maximum worker width=`32`.
- vLLM rejects simultaneous `moe_tp>1` and `moe_ep>1`.
- Disaggregate base pair counts before batch/worker enumeration: `AA=289`, `AB=68`, `BA=68`, `BB=16`, total=`441`.
- Pattern A and Pattern B cannot be one broad candidate bundle: it would introduce extra `dp>1, moe_ep=1, moe_tp=tp*dp` rows that do not represent the intended vLLM deployment semantics.

### Role and orchestration contract

- Aggregate Tasks use `agg_*_candidates`; disaggregate Tasks use independent `prefill_*_candidates` and `decode_*_candidates` fields, including explicit `*_cp_candidates=[1]`.
- Required experiment split: `agg_patternA`, `agg_patternB`, `disagg_AA`, `disagg_AB`, `disagg_BA`, and `disagg_BB`.
- The reference generated YAML resolves `num_gpu_per_replica` to `[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128]` and then applies `max_gpu_per_replica=64`. The Step4 plan records the explicit effective list `[1,2,4,8,16,24,32,40,48,56,64]` to avoid hidden values outside the total-GPU ceiling.
- Every disaggregate Task explicitly uses `max_gpu_per_replica=64`, `max_prefill_workers=64`, and `max_decode_workers=64`.
- `dp` is an intra-worker attention data-parallel width. `prefill/decode worker count` and replica allocation are deployment-level dimensions. They must be reported separately from `worker_gpus` and final `num_total_gpus`.

### Runtime OOM boundary

- Do not prefilter the 21 parallel rows based on expected memory use.
- AIC adjudicates memory feasibility per model/system/workload/runtime combination; a small worker becoming OOM at long ISL is a legitimate terminal outcome.
- Unknown exceptions must fail fast and must not be converted into an OOM label.
- OOM is also one cap-expansion terminal condition, but it is not permission to remove that parallel row before the run.

### Topology boundary

- `gb300` has `4` GPUs per physical node and `72` GPUs per rack; `intra_node_bw=900 GB/s`, `inter_node_bw=900 GB/s`, and `inter_rack_bw=100 GB/s` per GPU per direction.
- H100/H200 and test-side simulated H800 have `8` GPUs per node.
- `SystemSpec.get_p2p_bandwidth()` selects intra-node through the node limit, inter-node through the rack limit, and inter-rack above the rack limit.
- Therefore H100/H200/H800 groups `<=8` are intra-node and groups `>8` are inter-node. GB300 groups `<=4` are intra-node; groups `5..32` cross physical nodes but stay inside the NVLink rack and use `inter_node_bw=900 GB/s`. No accepted worker reaches the GB300 inter-rack tier.
- The reference statement “`tp<=8` is single-node” is valid for its 8-GPU-node systems but is false for `gb300`; `tp=8` crosses a physical node boundary there.

### Strict roofline-only correction boundary

- `DatabaseMode.SOL` selects the analytic op source, but current Task defaults can still alter the result after the SOL query:
  - `prefill_latency_correction=1.1` and `decode_latency_correction=1.08` multiply every operation in the disaggregate latency breakdown.
  - `rate_match_prefill_degradation=0.9` and `rate_match_decode_degradation=0.92` reduce disaggregate serving throughput.
  - `autoscale_ttft_correction_factor=1.8` scales prefill TTFT before SLA filtering.
- Primary comparison policy: explicitly set all five fields to `1.0` for both models and every Task. This preserves raw analytic latency, throughput, TTFT feasibility, and ranking inputs.
- Equal non-neutral factors are not automatically fair because SLA thresholds can prune each model differently and serving-mode penalties can alter aggregate-versus-disaggregate ranking.
- Independent StepCode Claude verdict=`APPROVE`; artifact: `.omx/artifacts/ask-claude-step4-roofline-corrections-20260713-0520.md`.
- Primary results are theoretical SOL/roofline estimates, not deployment predictions. Any future silicon-calibrated comparison must be a separately named experiment and must not be merged into the primary ranking.
## 2026-07-15 Formatter-Consistent Corrected-v3 Delivery Notes

- Ruff formatting changed the byte identity of the shared comparison runner after corrected-v2 execution. The v2 scientific payload was not numerically defective, but its strict execution-contract provenance no longer matched the final source bytes; the user approved new immutable v3 outputs rather than checkpoint rewriting.
- Primary v3 execution contract is `a13a4fe6ef9b932d01772ee3f0b8844760c52ec991fdcc5af641186baf1b697c`; full matrix-spec hash is `78b4970381ca7d0fb7bdfb53619a3280b72f936703c3c4d813bd23b96398edd4`; pinned Git HEAD is `9ce84ebbe3a0d7f785c91d055bdbdf4fdaabcbf1`.
- GB300 v3 finished naturally at `120/120`. v2/v3 scientific JSON is exactly identical after normalizing only `checkpoint_header.execution_contract_sha256`.
- The first GB300 disposable audit incorrectly used the full `480`-spec matrix hash for a `120`-spec shard and guessed `simulation_status=modeled`; the declared shard values are hash `79896333a080253b8f414ed7d67e316897b4d13f485b7e45b99a922863ce866f` and status `not_simulated`. The corrected full audit passed without artifact modification.
- The first merged semantic disposable audit compared a complete `14`-operation evidence Counter against a three-operation nested subset. v2 and v3 have identical complete Counters. The corrected gate checks the three required nested generation counts and separately requires outer `generation_moe_overlap=0`; the complete audit then passed.
- Fresh full unit evidence already matches the formatter-consistent source: `2002 passed`, `12 skipped`, `1119 deselected`, `4 warnings`, elapsed `765.21s`, exit `0`. The user explicitly requested minimum-cost completion, so no second full-unit run is permitted without a source/test change.
