## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-16 | Added the worker-1 Step4-Pro-V1 architecture mapping, independent arithmetic, provenance register, and implementation boundary audit. |

# Step4-Pro-V1 Architecture Mapping Audit

## Scope and source precedence

This is a read-only pre-implementation audit. The authoritative source is
`/data/ycfeng/stepfun-performance-optimization/permormancebenchmark/architecture_calculator_v1 - Main - latest.csv`
(SHA256 `f76fca03fd10eb145a04ff9ed906cdbd52beee37609103f9c99006e2bbf1920b`, 6423 bytes). The CSV explicitly supplies
Step4-Pro-V1 structure and arithmetic (`latest.csv:3-45`). Missing attention, MTP, quantization, latent-MLA,
and cache details are allowed to inherit the existing Step4 treatment, but each inherited value is listed below.
The report distinguishes **evidence** (direct source behavior or arithmetic) from **inference** (the proposed
configuration wiring or an approximation boundary).

## Evidence: CSV field to config, block, and operation mapping

The parser first reads generic `layers`, `hidden_size`, heads, vocabulary, context, and MoE fields
(`src/aiconfigurator/sdk/utils.py:612-642`) and then validates Step4-specific fields and materializes `Step4Config`
(`src/aiconfigurator/sdk/utils.py:787-852`). `Step4Model` consumes those values for its context and generation
operation graph (`src/aiconfigurator/sdk/models/step4.py:129-207,209-329,331-393,395-450`).

| Authoritative CSV field (Step4-Pro-V1 column) | Evidence / exact value | Config field and block implication | Operation implication |
|---|---|---|---|
| Input hint / attention heads | `swa 96 heads full 64 heads` (`latest.csv:3`) | `sliding_num_attention_heads=96`, `full_num_attention_heads=64`; these are validated into `Step4Config` (`utils.py:837-844`) | Audit labels remain `full` and `swa`; current temporary MLA graph uses `_num_heads` for both labels (`step4.py:151-156,190-196`), so this hint is not a faithful per-branch MLA geometry. |
| `model dim` / H | `6144` (`latest.csv:4`) | `hidden_size=6144`; returned as `model_info["hidden_size"]` (`utils.py:950-963`) and stored as `BaseModel._hidden_size` (`models/base.py:108-114`) | Residual width for attention projections, dense FFN, MoE router/experts, embedding, and logits (`step4.py:129-166,209-238,272-328,402-448`). |
| `layers` | `80`, explicitly trunk-only and excluding MTP (`latest.csv:5`) | `num_hidden_layers=80` → `_num_layers=80`; parser accepts explicit layer count (`utils.py:612-622`), base stores it (`models/base.py:108-114`) | Layer arithmetic and MTP scale use trunk depth; no extra MTP layers are included in CSV totals. |
| `full attn layers` | `20` (`latest.csv:6`) | Count `block_types.count("moe_full")=20` | `_layer_counts()` returns `full_count=20` and builds 20 full-label attention groups (`step4.py:122-127,395-407,423-436`). |
| `non-full attn layers` | `60` (`latest.csv:7`) | Aggregate SWA audit count is 60 | Normalized blocks must contain 4 `dense_swa` + 56 `moe_swa`; `_layer_counts()` counts dense blocks as SWA and returns `swa_count=4+56=60` (`step4.py:122-127`). |
| `num_dense_layer` | `4` (`latest.csv:8`) | Four leading `dense_swa` blocks; parser validates total length and legal labels, but does not enforce per-label counts (`utils.py:827-835`) | Dense FFN graph is instantiated with count 4 (`step4.py:395-409,423-438`); dense layers are also included in SWA audit count. Leading placement follows the documented Step4 grouping treatment (`common.py:251-259`; `task_memory/task_2026-07-10_step4_predefined_ops_plan/issues.md:148-154`). |
| `dense_inter_size` | `16384` (`latest.csv:9`) | `intermediate_size=16384` → `Step4Config.dense_inter_size` (`utils.py:819-825,850`) | Gated dense gate/up, SwiGLU, and down GEMMs use this width (`step4.py:209-238`). |
| `num routed` | `512` (`latest.csv:10`) | `n_routed_experts=512` → `model_info["num_experts"]` (`utils.py:633-642,960-962`) | Router output width and dispatch/MoE expert count are 512 (`step4.py:278-313,338-373`). |
| `num activated` / top-k | `8` (`latest.csv:11`) | `num_experts_per_tok=8` → `model_info["topk"]`; parser rejects `topk > num_experts` (`utils.py:633-642,834-835`) | Routed dispatch and gated MoE use top-k 8 (`step4.py:286-313,345-373`). |
| `latent moe size` | `0` (`latest.csv:12`), where zero means normal MoE formula | No latent-MoE field is consumed; retain normal routed-MoE treatment | `ops.MoE(... is_gated=True)` uses standard routed expert path (`step4.py:299-313,359-373`), not a latent-compression op. |
| `moe_inter_size` | `2048` (`latest.csv:13`) | `moe_intermediate_size=2048` → `_moe_inter_size` (`utils.py:641-642,962`) | Routed expert gate/up/down width is 2048 (`step4.py:299-313,359-373`). |
| `moe_shared_size` | `2048` (`latest.csv:14`) | `shared_expert_intermediate_size=2048` → `Step4Config.shared_expert_inter_size` (`utils.py:801,837-851`) | Shared gated FFN uses width 2048 and merges with routed output (`step4.py:241-270,327-328,388-392`). |
| `vocab size` | `128896` (`latest.csv:15`) | `vocab_size=128896` → `model_info["vocab"]` (`utils.py:623-626,950-963`) | Context/generation embedding and logits use vocabulary dimension 128896 (`step4.py:402-419,431-448`). |
| `HC.enable` | `0` (`latest.csv:16`) | No mHC parameter/op field is enabled | No mHC operation should be added; CSV mHC total is zero (`latest.csv:36`). |
| `HC.mult` | `0` (`latest.csv:17`) | No residual mHC multiplier | Same no-mHC conclusion; do not infer a hidden residual mixer. |
| `width-depth ratio` | `76.800` (`latest.csv:19`) | Derived check `6144 / 80 = 76.8` | Sanity check only; no independent op. |
| `num_moe_layer` | `76` (`latest.csv:20`) | Derived `80 - 4 = 76`; `Step4Model` computes `moe_count=self._num_layers-dense_count` (`step4.py:395-409,423-438`) | Context and generation MoE graphs are scaled by 76 layers. |
| `dense ratio` | `2.667` (`latest.csv:21`) | Derived `16384 / 6144 = 8/3` | Arithmetic sanity check for dense FFN width. |
| `param per dense` | `301,989,888` (`latest.csv:22`) | Derived `3*H*dense_inter_size` | Four dense layers contribute to stored parameters and active arithmetic. |
| `equiv inter width` | `18,432` (`latest.csv:23`) | Derived `moe_inter_size*topk + shared_size` | Active routed + shared FFN width for activation arithmetic. |
| `moe act ratio` | `3.0000` (`latest.csv:24`) | Derived `18,432 / 6,144` | Sanity check; no separate graph op. |
| `moe granularity` | `0.3333` (`latest.csv:25`) | Derived `2,048 / 6,144` | Sanity check; no separate graph op. |
| `moe all ratio` | `171.0000` (`latest.csv:26`) | Derived `(2048*512+2048)/6144` | Stored expert-width sanity check. |
| `activation per moe` | `342,884,352` (`latest.csv:27`) | Derived normal (latent size zero) active-MoE arithmetic | Gated MoE roofline workload is represented by `ops.MoE` (`step4.py:299-313,359-373`). |
| `param per moe` | `19,368,247,296` (`latest.csv:28`) | Derived all-expert stored-MoE arithmetic | Expert count/width are config metadata; runtime operation computes active top-k workload, not all stored weights. |
| `MoE flops GF/layer/token` | `2.057` (`latest.csv:29`) | Derived `6*activation_per_moe/1e9` | Arithmetic acceptance metric for each routed MoE layer. |
| `attn recipe` | `20 full + 60 nonfull` (`latest.csv:31`) | Preserve labels/counts in normalized block tuple | Full and SWA operation groups are named separately, but both use temporary MLA treatment (`step4.py:395-407,423-436`). |
| `param per full attn` | `153,095,232` (`latest.csv:32`) | CSV summary only; no complete projection recipe | Must be retained as an audit target, not reverse-engineered by invented operations. |
| `param per nonfull attn avg` | `213,911,648` (`latest.csv:33`) | CSV summary only; no complete projection recipe | Same audit-only treatment; temporary MLA is not claimed to close this value. |
| `attention total param` | `15,896,603,520` (`latest.csv:34`) | `20*full + 60*nonfull` weighted CSV total | Independent arithmetic target; current temporary MLA graph is not parameter-faithful. |
| `RMS` | `983,040` (`latest.csv:35`) | Derived `H*layers*2` | Layer-normalization parameter accounting only; graph uses norm elementwise operations. |
| `mHC total params` | `0` (`latest.csv:36`) | No mHC fields | No mHC op or parameter contribution. |
| `total param` | `1,489,092,340,608` (`latest.csv:38`) | Dense + stored MoE + attention + RMS (no mHC) | Acceptance arithmetic for trunk parameter inventory. |
| `total param w/ emb` | `1,490,676,214,656` (`latest.csv:39`) | Add `2*vocab*H` | Embedding/logits dimensions must use H=6144 and vocab=128896. |
| `total activation` | `43,164,756,864` (`latest.csv:40`) | Dense active + top-k MoE + attention + RMS | Acceptance arithmetic; not a measured latency. |
| `sparsity` | `2.90%` (`latest.csv:41`) | `total_activation / total_param` | Sanity metric only. |
| `attention activation ratio` | `36.83%` (`latest.csv:42`) | `attention_total / total_activation` | Sanity metric only; temporary MLA caveat applies. |
| `FFN activation ratio` | `63.17%` (`latest.csv:43`) | `(dense_active + moe_active) / total_activation` | Sanity metric only. |
| `KV cache @1M` | `10.7 GB` (`latest.csv:44`) | CSV cache target; detailed topology is not supplied | Must not be represented as proven by current temporary MLA formula; see mismatch below. |
| `KV + indexer cache @1M` | `10.7 GB` (`latest.csv:45`) | CSV cache target; indexer details are absent | Same human-update boundary; no inferred indexer op. |

## Evidence: normalized block composition

The only layer-level data in the CSV are aggregate counts. The minimum normalized tuple that closes those counts is:

```text
block_types = ("dense_swa",)*4 + ("moe_full",)*20 + ("moe_swa",)*56
```

This gives `4 + 20 + 56 = 80` trunk blocks, `20` full-attention MoE blocks, and `4 + 56 = 60` SWA-labeled
blocks. `_layer_counts()` intentionally counts `dense_swa` in the SWA total (`step4.py:122-127`). The ordering
(leading dense prefix, then full and SWA groups) is an **inference** inherited from the existing Step4 grouping
treatment, not a layer-by-layer fact in the CSV; `Step4Config` documents that it preserves audited classes without
claiming checkpoint order (`src/aiconfigurator/sdk/common.py:251-259`).

## Evidence: independent parameter arithmetic

Using `H=6144`, `L=80`, `full=20`, `swa=60`, `dense=4`, `moe=76`, `dense_inter=16384`, `experts=512`,
`topk=8`, `moe_inter=2048`, `shared_inter=2048`, and `vocab=128896`:

```text
param_per_dense = 3 * H * dense_inter
                 = 3 * 6144 * 16384
                 = 301,989,888

equiv_inter_width = topk * moe_inter + shared_inter
                   = 8 * 2048 + 2048
                   = 18,432

router_param = H * experts = 6144 * 512 = 3,145,728

activation_per_moe
  = router_param + 3 * H * equiv_inter_width
  = 3,145,728 + 3 * 6,144 * 18,432
  = 342,884,352

param_per_moe
  = router_param + 3 * H * (experts * moe_inter + shared_inter)
  = 3,145,728 + 3 * 6,144 * (512 * 2,048 + 2,048)
  = 19,368,247,296

attention_total
  = 20 * 153,095,232 + 60 * 213,911,648
  = 15,896,603,520

rms = 2 * H * L = 983,040

total_param
  = 4 * 301,989,888 + 76 * 19,368,247,296
    + 15,896,603,520 + 983,040
  = 1,489,092,340,608

embedding_param = 2 * vocab * H = 2 * 128,896 * 6,144
                = 1,583,874,048

total_param_with_embedding = 1,490,676,214,656

total_activation
  = 4 * 301,989,888 + 76 * 342,884,352
    + 15,896,603,520 + 983,040
  = 43,164,756,864
```

The values match CSV rows `22`, `27-29`, `32-35`, and `38-40` exactly (up to displayed rounding for GF and
percentages). Derived checks are `width/depth=76.8`, `dense_ratio=2.6666667`, `sparsity=2.898729%`,
`attention_activation_ratio=36.827738%`, and `FFN_activation_ratio=63.169985%`, matching CSV rows `19-21`
and `41-43`.

## Evidence versus inference: Step4-borrowed values

The following values are **not** supplied by the Step4-Pro-V1 CSV and must remain visible human-update items:

| Borrowed/approximate value | Evidence of source | Impact and boundary |
|---|---|---|
| Local identity `architectures=["Step4ForCausalLM"]`, `model_type="step4"`, ModelFamily `STEP4` | Existing config (`src/aiconfigurator/model_configs/stepfun-ai--Step4_config.json:2-10`), family mapping (`src/aiconfigurator/sdk/common.py:614-626`), and prior requirement (`task_memory/task_2026-07-10_step4_predefined_ops_plan/requirements.md:77`) | AIC local naming/dispatch contract; not proof of a public HF identity. New cached config should use `stepfun-ai--Step4-Pro-V1_config.json` and reuse `Step4ForCausalLM → STEP4 → Step4Model`. |
| Temporary MLA `num_heads=128`, `q_lora_rank=1536`, `kv_lora_rank=512`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128` | Existing Step4 config (`stepfun-ai--Step4_config.json:13,124-128`) and accepted temporary geometry (`task_memory/task_2026-07-10_step4_predefined_ops_plan/notes.md:262-280`) | Allows formula-only SOL/SOL_FULL MLA estimate. It is DeepSeek-V3-shaped borrowed geometry, not a verified Pro-V1 attention recipe. |
| Full/SWA KV heads `8/8`, attention head dim `128`, SWA window `512` | Existing Step4 config (`stepfun-ai--Step4_config.json:118-123`) | Query-head hints `64/96` come from Pro-V1 CSV, but KV heads/head dim/window are inherited. Current MLA graph does not consume the branch-specific query/KV fields or window (`step4.py:129-207`). |
| Generic `head_dim=128`, `num_key_value_heads=1` | Existing Step4 config (`stepfun-ai--Step4_config.json:5,16`) | Generic parser metadata is a placeholder for this MLA model; `get_kvcache_elements_per_token()` uses latent ranks instead (`step4.py:117-120`). Do not claim generic KV topology is Pro-V1 truth. |
| `num_nextn_predict_layers=3` (MTP) | Existing Step4 config (`stepfun-ai--Step4_config.json:17`); CSV excludes MTP (`latest.csv:5`) | Decode scale is inherited Step4 treatment. It is outside CSV trunk parameter totals and must be tested separately. |
| `quant_algo=fp8`, `quant_dynamic=true`, `kv_cache_quant_algo=fp8`, `torch_dtype=bfloat16` | Existing Step4 config (`stepfun-ai--Step4_config.json:18-22`) | Enables existing runtime quantization treatment. It is not a Pro-V1 CSV architecture fact; context FMHA compatibility may use BF16. |
| `hidden_act="silu"`, `max_position_embeddings=1048576` | Existing Step4 config (`stepfun-ai--Step4_config.json:6,9`) | Required metadata for cached parsing and 1M context scenarios; not supplied by architecture rows. |
| Block order (leading dense, then full/SWA MoE groups) | Existing normalized grouping policy (`common.py:251-259`; prior issue resolution `task_memory/task_2026-07-10_step4_predefined_ops_plan/issues.md:148-154`) | Aggregate counts close, but checkpoint layer-by-layer ordering remains unverified. |

## Important mismatch: attention and KV cache

### Attention parameter mismatch (blocking for faithful claims)

The CSV gives weighted targets `153,095,232` (full) and `213,911,648` (non-full average)
(`latest.csv:32-34`), but not the projection recipe. Applying borrowed Step4 GQA values (`KV heads=8`,
`head_dim=128`) yields `113,246,208` for full and `163,577,856` for SWA, leaving gaps of `39,849,024`
(`26.028913%`) and `50,333,792` (`23.530178%`). The source and root cause are documented in
`task_memory/task_2026-07-16_step4_pro_v1_support/notes.md:32-39` and `issues.md:11-18`.

**Inference / boundary:** Do not add hidden projections or a scaling factor. Preserve `20 Full / 60 SWA` labels,
use the explicitly temporary MLA graph, and mark all attention latency/parameter results as approximation-dominated
until a complete Pro-V1 Attention Detail source is supplied.

### KV-cache mismatch (blocking for faithful cache claims)

Current Step4Model computes latent MLA elements as
`num_layers * (kv_lora_rank + qk_rope_head_dim)` (`src/aiconfigurator/sdk/models/step4.py:117-120`). For
Pro-V1 borrowed ranks this is:

```text
80 * (512 + 64) = 46,080 FP8 elements/token
46,080 * 1,048,576 bytes = 48,318,627,840 bytes = 48.31838208 GB (decimal)
```

The CSV reports `10.7 GB` for both KV and KV+indexer at 1M (`latest.csv:44-45`), so the temporary formula is
`37.61838208 GB` higher and `4.515737x` the CSV metric. This is a topology/recipe discrepancy, not a calibration
factor opportunity. Report the CSV value as an external architecture target and the current formula as a temporary
implementation estimate; never claim they are equal.

## Findings by severity

### BLOCKING

1. **Faithful Full/SWA attention is blocked by missing projection detail.** The available CSV weighted totals cannot
   be reconstructed from the borrowed GQA/SWA geometry (`notes.md:32-39`). Temporary MLA may be used only with an
   explicit approximation label; invented hidden ops and scaling factors are prohibited.
2. **Faithful KV-cache capacity is blocked by topology mismatch.** Current latent-rank formula predicts
   `48.31838208 GB` versus CSV `10.7 GB` at 1M; this must remain a human-update item.

### IMPORTANT

1. `step4.py:136,140,147,182,186` still hard-codes temporary MLA widths `2112`, `24576`, and `32768`. They
   currently equal the borrowed formulas but prevent a second config from being data-driven. The minimal safe
   formulas are:
   - `2112 = q_lora_rank + kv_lora_rank + qk_rope_head_dim`;
   - `24576 = num_heads * (qk_nope_head_dim + qk_rope_head_dim)`;
   - `32768 = num_heads * (qk_nope_head_dim + v_head_dim)`.
2. Parser validation checks tuple length, legal labels, positive geometry, and `topk <= experts`, but not exact
   Pro-V1 counts (`utils.py:827-835`). A model-specific test must lock `dense=4`, `full=20`, `moe_swa=56`,
   aggregate SWA=60, and MoE=76.
3. Full/SWA query-head hints and windows are loaded into `Step4Config`, yet the temporary MLA operation graph
   consumes generic `_num_heads` and no SWA window (`step4.py:129-207`). This is intentional approximation, not
   evidence of faithful branch-specific latency.

### WATCH

1. Cached model identity must be wired through the existing loader naming convention (`utils.py:974-979`) and
   `DefaultHFModels` (`common.py:498-523`); network download must not become an implicit fallback.
2. Reusing `STEP4` family automatically retains the existing backend and database-mode guard (`step4.py:84-96`);
   adding a new model family or empirical perf database would widen scope and obscure the approximation boundary.
3. Any future attention-detail update must reconcile both weighted attention totals and the CSV cache rows before
   replacing the temporary MLA graph.

## Minimal recommended change set

1. Add only `src/aiconfigurator/model_configs/stepfun-ai--Step4-Pro-V1_config.json` with the exact CSV dimensions,
   block tuple `(dense_swa)*4 + (moe_full)*20 + (moe_swa)*56`, and explicitly documented inherited Step4 fields.
2. Add `stepfun-ai/Step4-Pro-V1` to `DefaultHFModels`; keep automatic cached loading and existing
   `Step4ForCausalLM → STEP4 → Step4Model` dispatch (`common.py:498-523,614-626`; `utils.py:974-979,1283-1289`).
3. Replace only the three hard-coded temporary MLA projection widths in `step4.py` with formulas derived from
   validated `Step4Config` fields. Preserve the original Step4 numeric shapes exactly.
4. Add RED/GREEN tests for cached resolution, exact block/count closure, all arithmetic above, borrowed-value
   provenance, projection-shape derivation, attention mismatch, KV mismatch, and original Step4 zero-regression.
5. Keep formula-only SOL/SOL_FULL operation behavior; do not add perfdb entries, empirical calibration, hidden
   attention operations, or fallback logic.

## Verification status

- **PASS:** authoritative CSV SHA256 and byte count match task notes.
- **PASS:** independent arithmetic matches CSV rows `19-43` (exact integers; displayed rounding for ratios/GF).
- **PASS:** source inspection covers cached loader, parser validation, family dispatch, block counting, MLA graph,
  and KV formula with exact line references.
- **N/A:** production typecheck, lint, unit, and e2e tests are intentionally not run because this worker report
  makes no source changes.

Subagent spawn evidence: 1 child probe (`repo_map_probe` / `requirements_probe`) was integrated for authoritative
CSV provenance, block-count closure, attention-gap arithmetic, and hard-coded MLA/KV risks.

Coordination protocol: coordinated - leader update was read back; task 2 is limited to this mapping report, while
roofline/integration report ownership remains with replacement tasks 3 and 6; no shared source or test files were
edited.
