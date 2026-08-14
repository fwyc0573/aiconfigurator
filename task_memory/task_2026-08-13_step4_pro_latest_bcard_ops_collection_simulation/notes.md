## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-13 | Recorded initial workspace, environment, and execution constraints. |
| 2026-08-13 | Added measured access facts: B300 quotagroup denial, the only grantable GPU path, Docker-via-sudo requirement, and shell cwd persistence. |
| 2026-08-13 | Revalidated B300 access and latest vLLM source/image identity after new platform evidence. |
| 2026-08-13 | Recorded user choice A: pinned local checkout and requirements shape are authoritative. |
| 2026-08-13 | Recorded the owner's no-action instruction for the temporary security-review file. |

# Operational Notes

- Repository: `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro`.
- Current branch: `step4-pro`.
- The worktree is already dirty with substantial changes from earlier tasks. No reset, bulk replacement, `rm`, or `mv` is authorized.
- This task has its own directory: `task_memory/task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation`.
- The local host is a CPU master unless `nvidia-smi` proves otherwise. GPU work must follow `/data/ycfeng/stepfun-env-handbook/guidence.md`; Docker work must follow `/data/ycfeng/stepfun-env-handbook/docker.md`.
- Large GPU allocations require a verified `rlaunch --predict-only` first and `--backoff-limit>0`.
- Do not write temporary files, logs, or caches to `/tmp`; use `/data/ycfeng/tmp`.
- Before Collector case changes, read `.agents/skills/aic-collector-op-development/SKILL.md`; before generator changes, read the applicable `.claude/rules/generator/**` rule.
- The referenced requirements document is `task_memory/step4pro_v4_external_simulator_requirements.md` and currently specifies B300 plus vLLM commit `607d1641ee3fec43653fca510d717725828890c2`.
- That requirements document links `step4pro_v4_shape_manifest.json` and `step4pro_v4_vllm_b300_8bit_task.md`, but those two files are not present at the expected `task_memory/` paths in the current checkout.
- The user supplied the source checkout path `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/vllm-step4-pro`.
- The supplied checkout is detached at commit `607d1641ee3fec436fca510d717725828890c2`, matching the requirements document's pinned commit.
- User decision on 2026-08-13: the requirements-pinned local checkout and 78-layer shape are authoritative; the later branch head is reference-only.
- The pinned checkout's two B300 smoke scripts both select image `hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled`; inspected manifest digest: `sha256:70492b0c79e2286b6ee56973f5f3322b53d293fc9332c4f792e56209a34d182b`.

## Measured access facts (2026-08-13)

- `nvidia-smi` confirms this host has no GPU. It is a CPU master.
- The earlier B300-denial conclusion is superseded. `b300_train_infra` is grantable and has passed single-node and multi-node predict-only checks; see resolved ISSUE-002.
- B300 is cross-zone. Launches must specify the pinned image, must not use `--volume`, and must move inputs/outputs through `tar | brainctl exec`.
- Docker requires `sudo`: the invoking user is not in the `docker` group, but `sudo -n docker` reaches the daemon. The `artifactory.stepfun-inc.com` mirror is already configured in `/etc/docker/daemon.json`, and `data-root` is `/data/var/lib/docker`.
- `hub.stepfun-inc.com` rejects unauthenticated catalog reads, so image tags cannot be enumerated from this host.
- Bash tool calls share a persistent working directory across invocations. Use absolute paths, or a stale `cd` from a previous call will make a relative path fail.
- Node listings from `brainctl get nodes` in plain-text form return a sample, not the full set. Use `-o json` with a label selector when an exact inventory is needed.
- Per explicit owner instruction, do not read, remove, alter, move, quote, or
  otherwise process
  `/data/ycfeng/tmp/aic_failure_domain1/codex_lane3_scope_baseline.md`.

## Revalidated runtime facts (2026-08-13)

- The vLLM repository is now reachable over SSH. The branch
  `xwx/step4pro-fa3-optimus` resolves to
  `9bfd9a610ea4f2890010702ee7a207cf25edf8de`.
- The requirements-document commit
  `607d1641ee3fec43653fca510d717725828890c2` is an ancestor of that branch
  head. The latest commit adds B300 DeepEP/NVSHMEM bootstrap changes.
- The branch scripts identify the B300 runtime image as
  `hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled`.
  Read-only registry inspection resolved manifest digest
  `sha256:70492b0c79e2286b6ee56973f5f3322b53d293fc9332c4f792e56209a34d182b`
  and config digest
  `sha256:48253062026862b2921f354613d3e9a7255e9540e3a7bf9148c9c80ec2d72661`.
- The source checkout contains `step4pro.py`, `optimus_fa4.py`,
  `optimus_fp8_moe.py`, and the Step4Pro model registry entry. Choice A makes
  this checkout, its declared B300 image, and the requirements' 78-layer shape
  the authoritative task contract.
- B300 predict-only succeeds with `b300_train_infra`; a 2-replica, 8-GPU
  per-replica request returned candidate B300 nodes. This supersedes the
  earlier incomplete quota probe.
