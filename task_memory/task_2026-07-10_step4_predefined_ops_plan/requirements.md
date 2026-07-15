## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-07-10 | Created requirements capture for Step4 predefined ops planning/grilling task |
| 2026-07-10 | Captured first grilling decision for Step4 non-full attention |
| 2026-07-12 | Captured AELLO Step4Air YAML review request and source-of-truth boundary |
| 2026-07-12 | Captured user decision to use gated SwiGLU for dense FFN |
| 2026-07-12 | Captured shared-expert FP8 projection and BF16 norm/merge decision |
| 2026-07-13 | Captured checkpoint nextn=3 with normalized nextn=0 comparison policy |
| 2026-07-13 | Captured temporary MLA substitution direction for unresolved attention modeling |
| 2026-07-13 | Captured all-92-layer MLA scope with retained Full/SWA audit labels |
| 2026-07-13 | Captured DeepSeek-V3 generic MLA geometry for the temporary substitute |
| 2026-07-13 | Captured local Step4 model ID, architecture, and ModelFamily naming contract |
| 2026-07-13 | Captured test-side simulated H800 SOL validation source |
| 2026-07-13 | Captured binary ISL interpretation for the requested `iso` matrix |
| 2026-07-13 | Captured primary-matrix OSL and independent decode-smoke policy |
| 2026-07-13 | Confirmed `step-design` as the later implementation branch |
| 2026-07-13 | Captured explicit `prefix=0` policy for comparison and decode workloads |
| 2026-07-13 | Captured validation coverage for both aggregated and disaggregated serving |
| 2026-07-13 | Captured reference-task TTFT-only SLA and ranking policy |
| 2026-07-13 | Captured aligned chunked-prefill-disabled matrix policy |
| 2026-07-13 | Captured the common 21-configuration parallel space and exhaustive batch-cap policy |
| 2026-07-13 | Captured the request to audit and repair the plan against the DeepSeek-V4-Pro reference search space |
| 2026-07-13 | Captured user authorization to execute the approved plan |
| 2026-07-13 | Captured acceptance of the recommended `OSL=1` Prefill-throughput ranking policy |
| 2026-07-14 | Captured explicit user adjudication approving ISSUE-046 remediation and continuation to the full matrix after verification |
| 2026-07-14 | Captured explicit user approval to execute the ISSUE-048 aggregate semantic remediation plan |
| 2026-07-14 | Captured option `1` approval for the latest independent-review BLOCK remediation |
| 2026-07-15 | Captured the requested branch review, archival commit, remote push, and `step-design` merge |
| 2026-07-15 | Captured approval for strict-provenance reruns and minimum-cost branch integration |

# Requirements: Step4 Predefined Ops Planning

## Raw User Intent

1. [Original Request] Use `$grill-me` for a new task in the `plan and discussion` stage.
2. [Original Request] Land the key docs now; do not execute business code modifications in this stage.
3. [Original Request] Use `permormancebenchmark/architecture_calculator_v1 - Main.csv` as the source of key model parameters for `step4` and `ds-v4-pro`.
4. [Original Request] AIC already has predefined ops for `ds-v4-pro`; implement equivalent predefined ops for `step4` in `aiconfigurator` on the intended branch `stepfun-design`.
5. [Original Request] Each op latency estimate for Step4 must use roofline modeling only; do not use profiling-based `perfdb` as the source of Step4 op latency estimates.
6. [Original Request] Follow the existing AIC style for predefined model ops by referencing how other predefined models define their op collections.
7. [Original Request] If current Step4 model parameters are ambiguous, reference HuggingFace Step3.7 model parameters.
8. [Original Request] After implementation, perform a sufficient validation matrix comparing Step4 with `ds-v4-pro`:
   - hardware: `gb300`, `h200`, `h100`, `h800`
   - sequence length setting: `iso=4k,16k,64k,256k,1m` as written by user; likely maps to AIC `isl`, pending confirmation
   - `total_gpu=64`
   - backend: `vllm`
   - `pp=1`
   - `tp<=8`
   - `ep` may be `>8`
   - GPU workers may be `>8`
9. [Original Request] Read and understand `permormancebenchmark/aello/examples/exp_suite/models/step4air_prefill.yaml` and `step4air_decode.yaml` to determine the dense FFN, non-dense/MoE FFN, full-attention, and non-full-attention algorithms, quantization, and implementation style.
10. [Original Request] If the currently planned full/non-full attention structure differs from those YAML files, correct the plan.
11. [Original Request] Treat the two YAML files as examples under one concrete parallel-sharding strategy only. Do not copy their layer counts, dimensions, expert counts, TP, or EP values into Step4 architecture; those values remain governed by `architecture_calculator_v1 - Main.csv`.
12. [Original Request] Compare the current Step4 plan with `task_memory/task_2026-07-04_aic_roofline_pareto_search/result-vllm-only-latest/todo_search_configs_deepseek-v4-pro.md`, identify any missing search-space dimensions, invariants, role mappings, or enumeration details, and repair the planning docs while preserving later explicit user decisions.
13. [Original Request] Confirm the approved plan and execute it to completion.
14. [Original Request] The user selected option `1`, explicitly approving the independently reviewed ISSUE-046 root-cause remediation, requiring TDD and review before continuing to the complete `480` mode-run matrix.
15. [Original Request] The user confirmed `确认，执行上述plan`, explicitly authorizing the ISSUE-048 aggregate mixed-step semantic remediation, associated review findings, required revalidation, and continuation toward the complete matrix only after all review gates pass.
16. [Original Request] The user selected option `1`, approving strict TDD remediation of the latest independent-review `BLOCK` before post-change full unit regression or the complete `480` mode-run matrix.
17. [Original Request] Review the current `task/step4-predefined-ops` worktree code and documentation, organize and archive the completed work in commits, push the current branch to the remote repository, then merge it into the sibling `aiconfigurator/` worktree's `step-design` branch and push the merged target branch.
18. [Original Request] Approve rerunning the current-source result shards needed to resolve strict execution-contract provenance mismatches while keeping GB300 background work running to natural completion.
19. [Original Request] Complete worktree cleanup, commit, push, and merge at minimum cost, avoiding unnecessary additional runs and duplicate test validation.

## Captured Decisions

1. [Original Request] Step4's `69 non-full attn layers` use GQA sliding-window attention with `96` query heads, `8` KV heads, `qk_head_dim=128`, `v_head_dim=128`, and `window_size=512`.
2. [Original Request] Step4's `23 full attn layers` use full GQA with `64` query heads, `8` KV heads, `qk_head_dim=128`, and `v_head_dim=128`; they are not the same head topology as the sliding-window layers.
3. [Original Request] Step4 architecture counts and dimensions come from the CSV, while the AELLO YAML files supply the requested attention/FFN algorithm, quantization, and implementation reference.
4. [Original Request] Dense FFN layers form a leading dense prefix; for the CSV Step4 architecture this means the first `4` trunk layers are dense and the remaining `88` trunk layers are MoE.
5. [Original Request] Step4 dense FFN uses gated SwiGLU rather than the omission-derived ungated AELLO default.
6. [Original Request] Step4's shared expert uses gated SwiGLU with FP8 gate/up/down projections, BF16 norm, and BF16 routed/shared output merge.
7. [Original Request] Step4's predefined checkpoint config declares `num_nextn_predict_layers=3`. The primary Step4-vs-DS-V4-Pro matrix explicitly sets `nextn=0` for both models, while Step4 `nextn=3` is validated separately with structural/smoke coverage.
8. [Original Request] Temporarily use MLA as the substitute for the unresolved Step4 attention modeling and explicitly declare that substitution wherever results are planned, validated, or reported.
9. [Original Request] Apply the temporary MLA SOL latency substitute to all `92` Step4 attention layers while retaining the `23 Full / 69 SWA` split as architecture and audit labels; do not claim that the shared substitute reproduces the real latency distinction.
10. [Original Request] Use the DeepSeek-V3 generic MLA geometry for the temporary substitute: `num_heads=128`, `q_lora_rank=1536`, `kv_lora_rank=512`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, and `v_head_dim=128`, while keeping Step4 `hidden_size=4096` and all Step4 layer/FFN/MoE parameters unchanged.
11. [Original Request] Use the local predefined model identity `stepfun-ai/Step4`, architecture `Step4ForCausalLM`, and ModelFamily `STEP4`; treat this as an AIC local naming contract rather than a verified public HuggingFace identifier.
12. [Original Request] Use the existing test-side simulated H800 system YAML through `--systems-paths default,tests/performance/aic_roofline_pareto/systems` for SOL-only validation, label H800 results as simulated, and do not add formal built-in H800 support in this task.
13. [Original Request] Interpret the requested `iso=4k,16k,64k,256k,1m` values as AIC input sequence length `isl`, using binary values `[4096, 16384, 65536, 262144, 1048576]`.
14. [Original Request] Use `osl=1` for every point in the five-point long-ISL Step4-vs-DS-V4-Pro primary matrix, and separately use `isl=4096, osl=1024` as a decode smoke point that covers generation ops without mixing long decode into the long-context matrix.
15. [Original Request] Use the currently active `step-design` branch for the later implementation; treat the original `stepfun-design` text as a branch-name typo.
16. [Original Request] Set `prefix=0` explicitly for every primary-matrix row and the independent decode smoke; do not include prefix-cache reuse coverage in this task's required matrix.
17. [Original Request] Run the Step4-vs-DS-V4-Pro validation matrix in both AIC aggregated (`agg`) and disaggregated (`disagg`) serving modes.
18. [Original Request] Reuse the SLA convention from `task_memory/task_2026-07-04_aic_roofline_pareto_search/result-vllm-only-latest/todo_search_configs_deepseek-v4-pro.md`: constrain only TTFT, do not constrain TPOT. Use TTFT values `[200, 500, 1000, 2000, 5000]` ms, represent non-binding TPOT as `50000` ms with `pareto_sweep=false`, and rank retained configurations by `tokens/s/gpu_cluster` as in that task.
19. [Original Request] Disable vLLM chunked prefill explicitly in both `agg` and `disagg` required matrix workloads. Do not add an aggregate-only chunked supplement and do not expand this task to fix disaggregated chunked-prefill plumbing.
20. [Original Request] Use the same exact `21` parallel configurations for Step4 and DeepSeek-V4-Pro: `17` pure-EP configurations with `moe_ep` in `[2,4,8,16,32]`, plus `4` pure-MoE-TP configurations with `tp=moe_tp` in `[1,2,4,8]`. Keep `pp=1`, `cp=1`, and `tp<=8`; include EP and worker sizes `16` and `32`; exclude `EP=64` from both models; do not add a DeepSeek-V4-Pro-only `EP=64` supplement; and do not expand this task to fix generic uneven-EP SOL modeling. For each serving path, start batch search at `1` with step `1`; use initial caps agg=`1024`, prefill=`16`, and decode=`1024`; double only a cap reached by the rank-1 result; continue until that result is below the cap, OOM, or TTFT-SLA-infeasible; and never treat a still cap-saturated row as final.
21. [Original Request] The user's confirmation accepts the immediately preceding recommended option for the five `OSL=1` primary points: keep the TTFT-only SLA filter and rank eligible configurations by cluster-normalized input/prefill token throughput rather than output-token throughput. Keep the `ISL=4096, OSL=1024` decode smoke on the existing output-token-throughput comparison path.

## Pending Grilling Questions

None. All required design decisions for this planning stage are resolved.
