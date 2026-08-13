## Modification History

| Date       | Summary of Changes                          |
|------------|---------------------------------------------|
| 2026-08-13 | Created the platform ticket packet for B300 quotagroup access. |

# B300 Quota Access Request — Evidence Packet

This document is written to be pasted into a platform ticket. Every claim below was
measured on 2026-08-13 from the CPU master `ycfeng`. No claim is inferred.

## 1. What is being requested

Add account `i-fengyicheng` (subject user ID `e7163316ad76ba7266a33f9787afd70a`,
project `shai/shai-core`) to a B300 quotagroup with at least **32 GPUs** of headroom.

The workload is a performance-modeling measurement campaign requiring three
configurations: 16 GPUs (one EP16 instance), 32 GPUs (one EP32 instance), and
32 GPUs (two EP16 instances).

## 2. Current symptom

Every attempt to reach B300 capacity fails. Two distinct failure modes appear,
depending on whether the request goes through a tag or through a quotagroup.

### 2.1 Tag-based selection returns no candidate

```bash
/kubebrain/rlaunch --predict-only \
  --charged-group=codesign --private-machine=group \
  --positive-tags=<TAG> \
  --gpu=8 --cpu=64 --memory=409600 --predict-node-num=5 -- bash -lc 'true'
```

| `<TAG>` | Result |
| --- | --- |
| `b300` | `no machine available` |
| `b200` | `no machine available` |
| `blackwell` | `no machine available` |
| `b300_sxm` | `no machine available` |
| `gb300` | `no machine available` |
| `b300-sxm6` | `no machine available` |
| `node/gpu-b300-0339.qy.cnw.istep.fun` | `no machine available` |
| `feature/GPUType=B300` | `no machine available` |
| `h800` (control) | **Succeeds** — returns multiple 8-GPU nodes with free capacity |

The `h800` control case proves the command form and the predict path are correct.
The B300 failures are access failures, not malformed requests.

### 2.2 Quotagroup-based selection returns HTTP 403

```bash
/kubebrain/rlaunch --predict-only \
  --charged-group=<QUOTAGROUP> --private-machine=group \
  --gpu=8 --cpu=64 --memory=409600 --predict-node-num=3 -- bash -lc 'true'
```

For `<QUOTAGROUP>` in `b300_sys_pro`, `b300_pretrain3`, and `b300_pretrain`, the
verbatim response is:

```text
fail to predict
code: 403 message: subject {Type:user ID:e7163316ad76ba7266a33f9787afd70a Name:} cannot get resource quotagroups/predict in API group quota.brainpp.cn/v1alpha1 in the project shai/shai-core reason:Forbidden
```

Listing quotagroups is also denied:

```text
Error from server (Forbidden): quotagroups.stepmind.com is forbidden: User "e7163316ad76ba7266a33f9787afd70a" cannot list resource "quotagroups" in API group "stepmind.com" in the namespace "default"
```

## 3. The hardware exists and is owned by quotagroups this account lacks

```bash
/kubebrain/brainctl get nodes -l GPUType=B300 -o json
```

Returns **20 nodes / 160 allocatable GPUs**. Ownership by
`privatemachine.brainpp.cn/quotagroup`:

| Quotagroup | Nodes |
| --- | ---: |
| `b300_pretrain3` | 11 |
| `b300_pretrain` | 5 |
| `b300_pretrain2` | 2 |
| `b300_sys_pro` | 1 |
| `b300_train_infra` | 1 |
| **Total** | **20** |

No B300 node carries the `codesign` quotagroup, which is the only quotagroup this
account can charge. Representative node labels, from
`brainctl describe node gpu-b300-0339.qy.cnw.istep.fun`:

```text
GPUType=B300
node.stepfun.com/gpu_cast_type=B300
node.stepfun.com/resource_pool=b300_pretrain3
privatemachine.brainpp.cn/project=shai-core
privatemachine.brainpp.cn/quotagroup=b300_pretrain3
privatemachine.brainpp.cn/tenant=shai
feature.brainpp.cn/nvidia-driver=580
rdma-network/link-type=RoCE
```

The account's project (`shai-core`) already matches the nodes' project. Only the
quotagroup membership is missing.

## 4. Why no substitute hardware works

The only grantable GPU capacity for this account is H800 via `codesign`. H800
cannot host this workload:

| Constraint | Value | Source |
| --- | ---: | --- |
| Static weights per GPU at EP16 | about 230 GB | Requirements document, section 2.4 |
| H800 memory capacity per GPU | 80 GB (85029158912 bytes) | `src/aiconfigurator/systems/h800_sxm.yaml` |
| B300 memory capacity per GPU | about 288 GB (288400343040 bytes) | `src/aiconfigurator/systems/b300_sxm.yaml` |

The model cannot be resident on H800 at the specified parallelism, so H800 is not
a degraded substitute. A prior task recorded the same reachability constraint at
`task_memory/task_2026-07-28_step4_pro_v3_v4_op_profiling/notes.md:34`, which
states that H100, H200, GB300, and Blackwell are not reachable from this account.

## 5. Environment facts, for triage

| Item | Value |
| --- | --- |
| Host | `ycfeng`, CPU master, no local GPU |
| CLI | `/kubebrain/rlaunch`, `/kubebrain/brainctl` |
| `brainctl version` | `v2.12.0-alpha.4.1260-20260805055936-171005b015dd` |
| Namespace | `shai-core` |
| Working quotagroup | `codesign` (H800 only) |
| Requested quotagroup | any `b300_*` with 32+ GPU headroom |

## 6. Secondary request, separable from the quota ticket

Two further access items block the same task but belong to different owners. They
are listed here so they can be routed rather than rediscovered.

1. **Git repository access.** `git@gitlab.basemind.com:sys/stepcast/vllm.git`,
   branch `xwx/step4pro-fa3-optimus`, commit
   `607d1641ee3fec43653fca510d717725828890c2`. The host is reachable (port 22
   open, HTTPS 302) but `git ls-remote` returns
   `Permission denied (publickey,password)`, and `~/.ssh/` contains no private
   key. An authorized SSH key or HTTPS token is needed.

2. **Container registry catalog access, or a correct image reference.** The
   locally cached image
   `hub.stepfun-inc.com/stepcast/stepcast:2026-07-14-server-vllm-0.19.0.post15-8a8f1b3f`
   does **not** contain the target implementation: `step4pro.py` and
   `optimus_fa4.py` are absent, and `registry.py:218` maps `Step4ForCausalLM` to
   the Step3p5 implementation. Catalog enumeration is denied
   (`UNAUTHORIZED` on the tags endpoint, HTTP 401 on `/v2/`), so the correct image
   cannot be located from this side. Either registry read credentials or the exact
   image reference with immutable digest would resolve this.
