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
| 2026-08-17 | Finalized the Phase 14 follow-up candidate with live evidence pull, one-delete teardown, strict cleanup/quota contracts, synchronized forward evidence, fresh `26/26` verification, and a `76`-file inventory. |
| 2026-08-17 | Recorded Phase 14 implementation commit `bd91ce3a`, exact 23-file staging, successful push, and local/tracking/remote SHA equality with ahead/behind `0/0`. |

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
two-node run exposed an independent EXIT-handler teardown defect. An
intermediate four-marker shutdown protocol repaired the first race and reached
`2/2` Running with rank-0 AgRs/DeepEP/automatic markers `1/0/0`, real
requests, and no `Broken pipe`, but that protocol is now historical.

The current Phase 14 lifecycle keeps both replicas and rank 0's shared
TCPStore alive after `remote_validation_ready`. The host pulls and validates
both evidence trees, requires
`DISTRIBUTED_RUNTIME_VALIDATION=PASS`, synchronized
`MODEL_FORWARD_COMPLETE.*batch=real` evidence on both replicas, clean logs,
DeepEP/automatic markers `0/0`, and at least one job-level
`AgRsAll2AllManager` marker. Only then does one exact
`brainctl delete rjob` own teardown. Cleanup fails unless both exact RJob and
Replica queries succeed and are empty.

Predict-only and live launch now reuse one argument array. Predict-only output
is archived and hashed, but it remains a per-worker fit check: replica counts
`2` and `8` returned the same seven candidate nodes. Live submission therefore
also requires independent disk-backed evidence of at least `16` available
B300 GPUs for `b300_train_infra`. Direct quota reads are RBAC-forbidden, and
the last trusted event reported only `6`, so Phase 14 submitted no live RJob.
The missing final two-node PASS remains an external admission blocker, not a
known DeepEP, NVSHMEM, AgRs, or communication-runtime failure.

The source-hash-bounded completion overlay calls
`current_platform.synchronize()` after model forward before logging
`MODEL_FORWARD_COMPLETE`. It proves CUDA completion and catches deferred
runtime failures without changing weights, communication selection, or the
training graph. The synchronize perturbs timing, so this smoke cannot be
reported as an uninstrumented performance benchmark. Fresh local verification
passes runtime contracts `26/26` in `0.21s`, shell syntax `6/6`, and Ruff
check/format on `5/5` Python files.

Phase 14 implementation commit
`bd91ce3a41fabde65b2e6f5707907a72b3ffb9a0` published the exact 23-file
task boundary. Local HEAD, the remote-tracking branch, and the direct remote
branch query matched that SHA immediately after push; ahead/behind was `0/0`.
This publication does not change the quota-blocked two-node live result.

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
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/test_report_2026-08-17_final_review_publication.md` | `0a55960b1ccf5e0dfa4e162ba3e485220aba6bb5a916a6baf5a63422d62af594` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/review.md` | `9bc9f376501c9033addc8367c3afccd58b9d3e994f47ce92693080ec1806e8ff` |

### Active Runtime Backend Requirement and Validation

| Path | SHA256 |
|---|---|
| `task_memory/step4pro_v4_external_simulator_requirements.md` | `356b47c624e2ef8bced3ad4db82c4cf1fa1e160b053c25b78ab5c89be9ac0c35` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/requirements.md` | `ff609b06dea91fee0552b2a624fdb372277d47c340224fd1469cab73ee6e2e6b` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/plan.md` | `92b624fcdbf8732b538accb68b00c8d28558bdbbd2bf48d586b520ec667a2912` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/harness.md` | `713ab5d647aa4bdf0a0abdc0dc33de140f79eb27a066834aff942df22762b9ff` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/design.md` | `27213c6df41a029b28c10587ba1d447c7c3e75a6ae9e5708cf4c0201fe97f92f` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/notes.md` | `87291a836ece28cbc219fe6bcdfb642aee64f0aefe090edb622ff6dc2c57298c` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/progress.md` | `d13343a56f614cde766ac545b71d94401563101938200994f9ede324797d1cee` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/issues.md` | `406673362a3ecb04f83e68557a21ece7682e1ee7547fd6e136647e309892570a` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/lessons.md` | `5bfa11bef7017d3545771be0e76d8e130d98fab6cc0266c75e34518727b6ee73` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/future.md` | `b2ee181def9dd6eef0574f537dad6ddfe3ccc18afc0657ab41fc44c1354287c9` |
| `tests/e2e/step4_pro_latest/b300_runtime_contract.sh` | `1a7ea63ac8bfc234719063ef0f92aae22bfbe5a89baa9ee2259268fa19000148` |
| `tests/e2e/step4_pro_latest/run_b300_single_smoke.sh` | `e761f317875168da99275f0a637774ba5901a9e74c406b53097a34dbc9bbc2f2` |
| `tests/e2e/step4_pro_latest/run_b300_two_node_smoke.sh` | `7a251a7878c4124403b901f779fdc22379077786a9fca2ac00397091f1f6a5b2` |
| `tests/e2e/step4_pro_latest/remote_b300_single_smoke.sh` | `c83422f93283b87f25beb6a21f8c4402c186e5ee013bc8a3e08b93291aeeca5d` |
| `tests/e2e/step4_pro_latest/run_b300_two_node_nccl_preflight.sh` | `775bea31ae418846260442d19051876c7139611f7bf323d41219a2e73b78ec54` |
| `tests/e2e/step4_pro_latest/run_b300_two_node_deepep_legacy_probe.sh` | `fe5a5a9e4a974914e4b250c025be96460bb16011eae057aba10dbee03c946963` |
| `tests/e2e/step4_pro_latest/test_b300_runtime_contract.py` | `d83c3048ded8590183b752c302457db4e952a58b031a87e5e8f702da3f3981f9` |
| `tests/e2e/step4_pro_latest/test_b300_single_smoke_contract.py` | `4604ebf968b4c619030ed50f939eb8c995cdbbb87caadddafb3cbf6d415a8867` |
| `tests/e2e/step4_pro_latest/test_b300_two_node_smoke_contract.py` | `cf25fa0437649933cb6e8903f0460de030dc4cc411767318c150654e4cf41d52` |
| `tests/e2e/step4_pro_latest/test_b300_two_node_nccl_preflight_contract.py` | `cf027fbb4cbbcb0ef46cb221e75c1d6ac206f10ce534c1e4f320e6460223d9d2` |
| `tests/e2e/step4_pro_latest/test_b300_two_node_deepep_legacy_probe_contract.py` | `8d018029612dd421b26ccc17e6155610831f1e1a3a5ec49ca7c510b27f9dac96` |
| `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation/test_report_2026-08-17_non_deepep_runtime_backend.md` | `1fad67857ed73c3b8d9bd5f51980795ac9b78324f73707b8518b349ce624ae7e` |

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
| Active AgRs runtime configuration | PASS | backend `allgather_reducescatter`; sequence parallel `0`; DeepEP/NVSHMEM package dependencies `0/0/0` |
| Active runtime contracts | PASS | `26/26` in `0.21s` |
| Active runtime source audit | PASS | manager `AgRsAll2AllManager`; dispatch/combine `all_gatherv`/`reduce_scatterv`; automatic backend selection allowed `0` |
| B300 predict-only | PASS_PER_WORKER_ONLY | initial single-node `4` candidates; latest two-node shape `7`; replica-8 control returned the same `7`; residual RJobs/Replicas `0/0` |
| B300 total-quota visibility | BLOCKED | direct quota read `Forbidden`; last trusted remainder `6`, required `16` |
| Phase 14 publication inventory | PASS | `76/76` recorded paths and SHA256 values match |
| Phase 14 implementation publication | PASS | commit `bd91ce3a`; staged files `23`; unrelated staged files `0`; local/tracking/remote SHA match; ahead/behind `0/0` |
| Single-B300 live smoke | PASS | scheduling `16s`; load `8.819692s`; request `13.737685s`; concurrent wall `0.474799647s` |
| Two-node AgRs live smoke | BLOCKED_BY_QUOTA | Historical coordinator run: `2/2` Running, rank-0 AgRs/DeepEP/automatic `1/0/0`, no `Broken pipe`; last live attempt: `0/2` replicas, request `16`, quota remainder `6`; Phase 14 live submissions `0` |
| Two-node cleanup | PASS (contract and prior evidence) | exactly one delete required; failed queries reject PASS; prior residual RJobs/Replicas `0/0` |
| Phase 14 focused runtime tests | PASS | `26/26` in `0.21s` |
| Published Phase 13 focused tests | PASS | `347/347` in `8.21s` |
| Published Phase 13 Collector tests | PASS | `401/401` in `31.60s` |
| Post-format reviewer contracts | PASS | `48/48` in `6.40s` |
| Phase 14 Ruff check/format | PASS | `5/5` Python files; findings `0` |
| Phase 14 shell syntax | PASS | `6/6` shell files |
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
   requires both validation-ready markers, both evidence trees pulled while
   the RJob remains live, synchronized real-batch completion on each replica,
   at least one job-level AgRs manager marker, DeepEP/automatic markers `0/0`,
   no `Traceback`, `ERROR`, or `Broken pipe`, exactly one RJob delete, and
   successful empty exact-name cleanup queries.
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
   superseded shutdown-marker lifecycle, require a globally scoped manager
   line from every replica, treat predict-only as total-quota proof, or add
   sleeps/blind retries.
