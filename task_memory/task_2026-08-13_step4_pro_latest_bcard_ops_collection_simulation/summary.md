## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-13 | Created the completion-summary scaffold; final inventory is pending execution. |
| 2026-08-15 | Replaced the initial scaffold with the current AIC implementation, B300 dataset, and blocked-simulation checkpoint. |
| 2026-08-15 | Added Full-MFA QKV B300 data and refreshed all hashes, coverage metrics, simulation values, and remaining gaps. |
| 2026-08-15 | Added the final formatting-only repair, fail-fast validation evidence, and refreshed affected hashes. |
| 2026-08-15 | Recorded the active SWA QKV expected-RED gate and refreshed the provider-core contract hash. |
| 2026-08-16 | Finalized the approved MTP-off archive after complete QKV collection, independent review, and final verification. |
| 2026-08-16 | Removed a duplicated word from the future DeepEP completion condition. |
| 2026-08-16 | Finalized Phase 10 with the explicit DeepEP proxy, 235-row Attention dataset, full proxy matrix, and refreshed inventory. |
| 2026-08-16 | Refreshed the review-record hash after removing one duplicated Phase 10 archive entry. |
| 2026-08-16 | Finalized Phase 11 with the requirement audit, three-repeat/manual-review artifacts, fresh verification, and refreshed hashes. |
| 2026-08-16 | Refreshed the final publication evidence, regression timings, and affected inventory hashes. |
| 2026-08-16 | Normalized the review CSV to LF-only output and refreshed final verification evidence and inventory hashes. |
| 2026-08-17 | Added Phase 12: replaced active DeepEP runtime settings with explicit AgRs/NCCL communication, archived its contracts, and refreshed the deliverable inventory. |
| 2026-08-17 | Added predict-only, single-B300 PASS, rank-0 AgRs execution, and the blocking two-node lifecycle result. |
| 2026-08-17 | Synchronized the completed local coordinator/global-marker-scope fixes, fresh `14/14` contracts, and the external 16-GPU quota blocker. |
| 2026-08-17 | Corrected the final admission rule after proving predict-only is per-worker only and direct quota reads are RBAC-blocked. |
| 2026-08-17 | Revalidated the stable post-concurrency runtime files, refreshed stale hashes, and corrected the Phase 12 inventory count to `16/16`. |
| 2026-08-17 | Completed the Phase 13 code/docs review, normalized the current and historical runtime reports, refreshed `347/347` and `401/401` evidence, and expanded the publication inventory to `70/70`. |

# Summary

## Task Overview

The approved MTP-off `stepfun-ai/Step4-Pro-Latest` AIC graph, Collector
contracts, B300 datasets, and scheduler-aware simulation driver are complete
against pinned vLLM commit
`607d1641ee3fec43653fca510d717725828890c2`.

Fresh B300 measurement is complete for every owner-approved measurable family:
Attention `235/235`, Optimus MoE `174/174`, grouped `wo_a` `75/75`, FP32
router `75/75`, Full-MFA QKV `75/75`, and SWA QKV `75/75`. The six canonical
files contain `709` rows. Historical H800 rows were not used.

DeepEP HT remains defined and fail-fast consumed, while exact measurement is
explicitly deferred at `0/116`. The owner-approved, simulation-only B300 NCCL
`alltoall` proxy completed all `72` prefill and `84` decode records with
`result_fidelity=PROXY`: prefill latency is
`129.22378441076575–782420.0031436655 ms`, and decode batch-1 TPOT is
`56.955545921130614–268.6990437660492 ms`. No fake DeepEP Parquet was
created. MTP1 remains deferred because the pinned Step4Pro source has no native
MTP1 implementation.

The active whole-model runtime requirement no longer depends on DeepEP. The
single-node and two-node wrappers now pin
`allgather_reducescatter`/`AgRsAll2AllManager`, set sequence parallelism to
`0`, reject every DeepEP manager variant and automatic backend selection, and
remove explicit active NVSHMEM and `deep_ep` evidence dependencies. The pinned
implementation uses NCCL-backed `all_gatherv` for dispatch and
`reduce_scatterv` for combine. This AgRs runtime is distinct from both the
exact AIC DeepEP operation identity and the completed NCCL `alltoall`
simulation proxy.

Both exact resource shapes passed the initial B300 predict-only checks. The
single-B300 smoke passed with scheduling `16s`, model load `8.819692s`, first
request `13.737685s`, and four-request wall time `0.474799647s`. The original
two-node run exposed an independent EXIT-handler teardown defect; the approved
root fix now uses validation-ready, shutdown-arm, armed-acknowledgement, and
concurrent final-shutdown stages. Its local contracts pass `14/14`. The first
coordinator live run reached `2/2` Running, recorded rank-0 AgRs/DeepEP/auto
markers `1/0/0`, completed real requests, and showed no `Broken pipe`; its
rank-1 gate was then corrected because the manager line is logged once with
`scope="global"`. The final corrected payload created `0/2` replicas because
it requested `16` B300 GPUs while the queue had only `6` remaining. Cleanup
left `0` RJobs and `0` Replicas. The missing final two-node PASS is therefore
an external admission blocker, not a known DeepEP, NVSHMEM, AgRs, or
communication-runtime failure. A fresh diagnostic returned the same seven
per-worker candidates for replica counts `2` and `8`, proving predict-only
does not validate total replica quota. Direct quota reads are RBAC-forbidden,
so no new live run was submitted.

The post-concurrency stability check found that the remote runner and its
two-node contract were written after the preceding summary inventory. Their
current contents remain within the approved coordinator design and pass the
combined runtime contracts `14/14` in `0.04s`. The refreshed active-runtime
inventory now matches `16/16`; the earlier `17/17` statement was a count
error, not an additional missing deliverable.

The final requirement audit passed for the approved AIC scope. It checked all
`156` required cases, the LF-only `156 × 87` manual-review table, `468` repeated case
executions, component and KV accounting, all `709` canonical B300 rows, and
DeepEP-Parquet absence. A fresh complete run produced the same
`b68008b8...dd61a` SHA256 as the archived full result. Live router
observations, an AgRs-specific simulator model, and the vLLM-versus-AIC error
table remain unavailable and were not fabricated.

Compared with legacy H800 V3/V4 work, Latest preserves the pinned mixed
20-layer Full-MFA plus 58-layer SWA graph, shared-KV physical layout, grouped
`wo_a`, FP32 routing, Optimus MoE, and DeepEP HT identities. The independent
provider-core reviewer returned **APPROVE** with no remaining Blocking or
Important finding. Phase 10 received a complete local audit; it did not receive
a separate independent sub-agent review.

## Deliverables Inventory

### Core Implementation and Provenance

| Path | SHA256 |
|---|---|
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/step4_pro_latest_shape_manifest.reconstructed.json` | `c3089c53e2e711e73defbc24aeda2e1df3236047f67b8717146235778528fa9e` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/op_provenance.md` | `3b3faa9d876ef6ecb36873ddd99ccc79997bd527a274eeca50ab8430bb58fada` |
| `collector/case_generator.py` | `97116db78345dc1658f3350a1cfaa21b38ff4492744c1ee189623447f1fc7b8f` |
| `collector/cases/models/Step4ProForCausalLM_cases.yaml` | `6b6ae15de96e63047f324b93321058800126c9939bb755f7b15055b52370d75e` |
| `collector/framework_manifest.yaml` | `3247a9eadd9795ed9e5bf168078d7e452dfb7291201f22946e5d98f278268aa2` |
| `collector/registry_types.py` | `de15bb0d5a281f410c1c61b812f841ff39d4a2931db761679831f341064ef94a` |
| `collector/vllm/collect_step4_provider.py` | `6ef090963679d233218c7799ea1acaf372aa67ca11ce6135cb4313b49a961b60` |
| `collector/vllm/registry.py` | `a0a260d7792eddfc44a507835aab253cd1085be4aa0057de77848aa616b891be` |
| `collector/wideep/vllm/collect_step4_deepep_ht.py` | `97c8b17a76b1f83a45de3bde20b6d7c6e31ea89f49eaae0536ef2c8afb31be31` |
| `collector/wideep/vllm/registry.py` | `b09ae5b930975f7a4ad9e6a2f70b82f0ac9b6ada808f0f3e88418da073ae09b6` |
| `src/aiconfigurator/sdk/backends/base_backend.py` | `d8df3fa87999daeb926c5c13295faec068a23269f037e0ad908b35478d1e9c71` |
| `src/aiconfigurator/sdk/common.py` | `e5ce3b28b6040bb0ff604d84ab486351db0a2137e1dd9eeba973bdbfe5451f8c` |
| `src/aiconfigurator/sdk/config.py` | `9aa0b57953544734fcd5c7def15b2a399ad95f0a6e4272914d6779414096fae6` |
| `src/aiconfigurator/sdk/models/base.py` | `b1101219dd754a81c14b70ffd068d708eb4116e62088ac6af5f8122cc008bc22` |
| `src/aiconfigurator/sdk/models/step4.py` | `8bb428227de65119ea4bc16cf674164363db6aa334db1552c2c3cb1dc580376f` |
| `src/aiconfigurator/sdk/operations/moe.py` | `beb280df3fc014bd7a0f45655354e453ae3e53600873c6705d5e42f5a99aed2c` |
| `src/aiconfigurator/sdk/perf_database.py` | `f01a053c521dc783ffbbee0c484e781f84ff9aaebb80547731d9bc4154f71824` |
| `tests/performance/step4_pro_latest/run_b300_attention_collection.sh` | `6781811bfd93d33dde2ccbd32dfe2bc928d6fa46ac56008cdffbc5e19ff92789` |
| `tests/performance/step4_pro_latest/remote_b300_attention_collection.sh` | `08234d9e16593187755d0d34482106977c2fd0e5c266d687c06557213abe7dd4` |
| `tests/performance/step4_pro_latest/run_b300_provider_core_collection.sh` | `ea678eb3ba21838893e1eaf114a73c9416b49e0c92ee29e1225f6db1f8637be6` |
| `tests/performance/step4_pro_latest/run_b300_optimus_moe_collection.sh` | `be11fd9b891fe7ec2c1f24c4d036bc6f7b557433e515d2d616ee02c4b58ce470` |
| `tests/performance/step4_pro_latest/run_b300_deepep_ht_collection.sh` | `c986c7126e9022454399d326c93c1022267179d0b9d08c39cef17b4c300ef3f3` |
| `tests/performance/step4_pro_latest/validate_b300_provider_core_rows.py` | `34765c21487c599eb0fda55ea5d2edbadf2cbe1540ccc27bbdd8eda615cd7067` |
| `tests/performance/step4_pro_latest/deepep_proxy.py` | `5cc29a804b15f178361890fb5518d4b6b536bf6f57ce5eab618ad04c3119f032` |
| `tests/performance/step4_pro_latest/run_mtp_off_requirements.py` | `de8a81eabaa3f5c216cfef57d10f1563463a4773d21aeb28cd6058d1b6e4fe6a` |
| `tests/performance/step4_pro_latest/validate_aic_silicon_coverage.py` | `bbd2745c2ce55b27996869f8231b7055af80f0f2758b1381ad1dcfa3b0fbab6e` |
| `tests/unit/performance/test_step4_pro_latest_deepep_proxy.py` | `1bd0f23a0233987276fec1af72f87d1dd78ba323cfdaab1bc611064a78b73256` |
| `tests/unit/performance/test_step4_pro_latest_mtp_off_requirements.py` | `4c839de471fc1759d524a03ce92a75e4cd9d9115f16802639a4f1b65972e45ea` |
| `tests/unit/performance/test_step4_pro_latest_silicon_coverage.py` | `e1eecdab879bdb17cc6a8bb7d4cc661481f2876bd96b686fc90a8e8ace92548c` |

### Canonical B300 Data

| Path | Rows | SHA256 |
|---|---:|---|
| `src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0/step4_context_attention_perf.parquet` | 68 | `7cf90e1f508e9f18dceb29c51dc90e88caa1f13d3d25d38f4a3435851c79522a` |
| `src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0/step4_generation_attention_perf.parquet` | 167 | `bde836b884b410b21284645ab45f0a35f82d03cac5b91299b256823067cb14c0` |
| `src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0/step4_optimus_moe_perf.parquet` | 174 | `4bfb1ccdfa8007d3a23576b4e7d50e10dbb11fef6500db29668a59f530cad388` |
| `src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0/step4_grouped_gemm_perf.parquet` | 75 | `78fdd68077e3fdaa4c4fa349ab0a72407e0421ae09ecfcfcd4a5ae22d103d760` |
| `src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0/step4_fp32_output_gemm_perf.parquet` | 75 | `3ec22771fa8577dec8b250ca1d6152552a8091b9502c1732461df65f5dc97af9` |
| `src/aiconfigurator/systems/data/b300_sxm/vllm/0.19.0/step4_qkv_norm_rope_perf.parquet` | 150 | `cc14553d8e0d619ac5fc21b15a9dd78633e168114da270e55edec2c6e7afd579` |

### Validation and Simulation Evidence

| Path | SHA256 |
|---|---|
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/qkv_b300_canonical_consumer_validation_2026-08-16.json` | `0a45e1655b9b5bf88710aba8a95a7056ebfa97888e78a654f1e5191265042758` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/aic_silicon_coverage_2026-08-16.json` | `00d4c46fc068867c7d480dca03e19c0c0ed867c217a62e07187061ecc461bc25` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/mtp_off_requirements_exact_smoke_2026-08-16.json` | `142ec21e10e0adb5132e011f058166196539918e4d85d0d6710b074b99e3e82d` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/mtp_off_requirements_full_2026-08-16.json` | `f2b3f8f04c4e966c48bb55277f33a8e8b3d243062efd1674c1eb3687ca3cb7de` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/aic_silicon_coverage_proxy_2026-08-16.json` | `0b5f19fa558970a6d8940052521b0e68fb62c301ded833881b9a0fbc27038285` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/mtp_off_requirements_proxy_smoke_2026-08-16.json` | `d270ae17b0176eda4baf88266d3f803d2e51ae88564a96613a9d4fc5209dac48` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/mtp_off_requirements_proxy_full_2026-08-16.json` | `b68008b8a86c0a67b90a51fc9b57c301c7561101a9e3613b405e75c6482dd61a` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/mtp_off_requirements_proxy_repeat3_2026-08-16.json` | `9d5c296c7c4e95859982cbb986cbda21ef65f9a56d286ebe37cc5a780f7208a1` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/mtp_off_requirements_proxy_review_2026-08-16.csv` | `4c1afd34a37877a6cb59cf79c37a326b537261ed9640ee4d657ce527ff49ac62` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/test_report_2026-08-15_step4_pro_latest_attention.md` | `6ca67467d556715034f85b03ef9d412fb5758b04dfd3c135416aa9eb90b5ee15` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/test_report_2026-08-15_step4_pro_latest_optimus_moe.md` | `eed859963a7b23a3d3269f2cd7293dbbc1db02999c2190493609bc4bc288ecbc` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/test_report_2026-08-15_step4_pro_latest_grouped_gemm.md` | `51876eb59d544e96babe87445630394881a501be19b91802e9c6f4b5ad5dd129` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/test_report_2026-08-15_step4_pro_latest_fp32_router.md` | `fc469642cc733235459bb3e2eb17e5f42608beaf7cf72c505a56f5d712d421d4` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/test_report_2026-08-15_step4_pro_latest_qkv_norm_rope.md` | `9ae9be7d30745684a8100f7b0979cb7a7c030f45bc0bc7320760835352e09a7f` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/test_report_2026-08-15_step4_pro_latest_deepep_local_ipc.md` | `cfb88a6a8d51234717954937dba9f5a1471b2a383a488c85bdbcb4957399702b` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/test_report_2026-08-15_step4_pro_latest_mtp_off_simulation.md` | `0bca3b9075d884912d7faf4251a65e41d558d08730265be760fd1d14d68fec2e` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/test_report_2026-08-14_b300_pinned_vllm_smoke_runtime_trace.md` | `c22115b87481eaa9ae41131bb0f01772ea2c98d6b16a3eb4e5755af54270afb6` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/test_report_2026-08-17_final_review_publication.md` | `5825e4f5b759891975f1d0761dc60797aa660730eaef331f129bbe898df24691` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/review.md` | `4d5926380ab15b549f454c23ebf0445b9de207bdecaafc06707e0b3919474e69` |

### Active Runtime Backend Requirement and Validation

| Path | SHA256 |
|---|---|
| `task_memory/step4pro_v4_external_simulator_requirements.md` | `356b47c624e2ef8bced3ad4db82c4cf1fa1e160b053c25b78ab5c89be9ac0c35` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/requirements.md` | `ff609b06dea91fee0552b2a624fdb372277d47c340224fd1469cab73ee6e2e6b` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/plan.md` | `9e61dfcbd74df769de150770ce41e2931946057f087a13ed4e918ae385197fa6` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/harness.md` | `00cb24d06509d0d2d74cd0db6ebc12ca9e3c1504167e110b96da8f634586894a` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/design.md` | `5d403f1888cd5ad10cb38039acd35214d392ec3d8ed1c79bcd54706cba51d8f5` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/notes.md` | `4c2b37d0d38b63ea94d15f80eab1b9cd9f552ce39a10298db07a82f5e391c716` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/progress.md` | `6f58014719b91e300a1cdc0b9a17076ee175cfac3791cf1dd69d80c4a4ed8607` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/issues.md` | `b09c6af6f8d25bb31bc483eef2eaedb22cd364e39f9933341de56ad2cfc0be55` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/lessons.md` | `1bf4e07c6f9cdad73a95e37a5fe56e3a52124c8b9492ee70ac63d21c34759bf5` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/future.md` | `ffd019fccf78c700ca9830f9e8c1a63cb7ffdb8518a4fe9f907161edf40d36bb` |
| `tests/e2e/step4_pro_latest/run_b300_single_smoke.sh` | `47bc6cf8507281ade56f0d1e96913f75386e6b7d30f8212a7e8f9f5012375e1d` |
| `tests/e2e/step4_pro_latest/run_b300_two_node_smoke.sh` | `22ab5eb03cba75f589b36d73e95af9bfcd6617af86128ebefdc9121a59115cd5` |
| `tests/e2e/step4_pro_latest/remote_b300_single_smoke.sh` | `7cc139aab85d3db29e53e2b750fd1d302d82d00a84a706368524e041d5d13e15` |
| `tests/e2e/step4_pro_latest/test_b300_single_smoke_contract.py` | `390c1c74517e921099f85d776e6346473ccc3ad573aff80eda9a4c0695f093f5` |
| `tests/e2e/step4_pro_latest/test_b300_two_node_smoke_contract.py` | `9b749a806fb5f60e78dd7b4a32967bfaaee14d25b39452ba776579f112f166c6` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/test_report_2026-08-17_non_deepep_runtime_backend.md` | `638484ff7720994cb077f49be07c9a42de37582d4e7e2439e6ac33ed480960b6` |

## Validation Status

| Area | Status | Numeric Result |
|---|---|---|
| MTP-off model/op graph | PASS | Context and generation each contain `1,821` ordered operations |
| Corrected Attention dataset | PASS | `235/235` exact silicon queries; scheduler `394/394`; max error `0.0 ms` |
| Optimus MoE dataset | PASS | `174/174` exact silicon queries |
| Grouped/router datasets | PASS | `150/150` exact silicon queries |
| Combined QKV dataset | PASS | `150/150`, providers `75+75`, max error `0.0 ms` |
| DeepEP HT dataset | DEFERRED | `0/116`; no substitute or historical data |
| Canonical B300 dataset | PASS | `709` rows across six Parquet files |
| Active AgRs runtime configuration | PASS | `3/3` scripts; backend `allgather_reducescatter`; sequence parallel `0` |
| Active runtime contracts | PASS | `14/14` in `0.04s`; DeepEP/NVSHMEM/package dependencies `0/0/0` |
| Active runtime source audit | PASS | manager `AgRsAll2AllManager`; dispatch/combine `all_gatherv`/`reduce_scatterv`; automatic backend selection allowed `0` |
| B300 predict-only | PASS_PER_WORKER_ONLY | initial single-node `4` candidates; latest two-node shape `7`; replica-8 control returned the same `7`; residual RJobs/Replicas `0/0` |
| B300 total-quota visibility | BLOCKED | direct quota read `Forbidden`; last trusted remainder `6`, required `16` |
| Phase 12 inventory hash audit | PASS | `16/16` current files match the recorded SHA256 values |
| Phase 13 publication inventory | PASS | `70/70` recorded paths and SHA256 values match |
| Single-B300 live smoke | PASS | scheduling `16s`; load `8.819692s`; request `13.737685s`; concurrent wall `0.474799647s` |
| Two-node AgRs live smoke | BLOCKED_BY_QUOTA | First coordinator run: `2/2` Running, rank-0 AgRs/DeepEP/automatic `1/0/0`, no `Broken pipe`; final corrected run: `0/2` replicas, request `16`, quota remainder `6` |
| Two-node cleanup | PASS | final exact-name RJobs/Replicas `0/0` |
| Final focused tests | PASS | `347/347` in `8.21s` |
| Full Collector tests | PASS | `401/401` in `31.60s` |
| Post-format reviewer contracts | PASS | `48/48` in `6.40s` |
| Ruff check/format | PASS | `45/45` task-owned Python files |
| Shell syntax | PASS | `14/14` scripts |
| Structured inputs | PASS | YAML `2/2`; JSON `3/3` |
| Whitespace | PASS | `git diff --check` exit `0` |
| Independent provider-core review | APPROVE | `0` remaining Blocking/Important findings |
| Phase 10 local audit | APPROVE | No orphan helper/schema, no no-op deduplication, no unrelated orchestration change |
| Phase 11 local audit | APPROVE | No uncovered condition in the approved MTP-off plus explicit-proxy scope |
| Proxy coverage audit | PASS_WITH_PROXY | `36,420` records; `18,220` silicon; `15,160` non-silicon; `3,040` proxy; missing/error `0/0` |
| Requirement artifact audit | PASS | JSON `156` cases; CSV `156 × 87`; component error `6.51925802230835e-09 ms`; KV error `0.0 GiB` |
| Review CSV serialization | PASS | LF `157`; CRLF `0`; bare CR `0` |
| Three-repeat audit | PASS | `468` executions; `156/156` identical; maximum spread `0.0` |
| Fresh full simulation | PASS_WITH_PROXY | `72 + 84`; byte-identical SHA256 `b68008b8...dd61a` |
| Exact DeepEP contracts | DEFERRED | Dispatch/combine at EP16 and EP32; dataset `0/116` |
| Prefill proxy matrix | PASS_WITH_PROXY | `72/72`; latency `129.22378441076575–782420.0031436655 ms`; OOM `48/72` |
| Prefill proxy throughput | PASS_WITH_PROXY | per replica `1340.1293369125042–69183.9264811136 token/s`; aggregate `1340.1293369125042–103512.27987036602 token/s` |
| Decode proxy matrix | PASS_WITH_PROXY | `84/84`; batch-1 TPOT `56.955545921130614–268.6990437660492 ms`; HBM `138.58154296875–294.77685546875 GiB` |
| Decode proxy capacity | PASS_WITH_PROXY | `B_max=0`, `aggregate_B_max=0`, `first_failed_batch=1` for `84/84` |
| Proxy persistence isolation | PASS | `step4_deepep_ht_perf.parquet` absent |
| H800/V3/V4 isolation | PASS | `0` historical H800 rows used as B300 results |
| MTP1 | DEFERRED | No native pinned Step4Pro MTP1 path |

## Open Items/Future Extensions

1. Obtain direct platform or quota-owner evidence that the B300 queue can
   admit at least `16` GPUs. Then rerun the same-shape predict-only check for
   per-worker fit and exactly one corrected two-node live wrapper. Acceptance
   requires both validation-ready markers, both shutdown-arm acknowledgements,
   at least one job-level AgRs manager marker, a real-batch marker on each
   replica, DeepEP/automatic markers `0/0`, no `Broken pipe`, and cleanup
   `0/0`.
2. Add or calibrate an AgRs-specific AIC communication model before reporting
   a same-backend vLLM-versus-simulator communication error. The existing
   NCCL `alltoall` result remains `PROXY`.
3. DeepEP EP16/EP32 measurement is now optional comparison work rather than an
   active-runtime prerequisite. If resumed, it still requires a working
   NVSHMEM launcher/runtime and must replace proxy results before exact
   DeepEP latency or capacity is reported.
4. MTP1 structure, measurement, and simulation remain deferred until an
   authoritative Step4Pro implementation is provided.
5. Preserve the failed and quota-blocked two-node evidence. Do not restore the
   independent one-node EXIT lifecycle, require a globally scoped manager line
   from every replica, or add sleeps/blind retries.
