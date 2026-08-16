## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-15 | Recorded the B300 eight-rank DeepEP local Buffer discriminator and blocking result. |

# Test Report: Step4-Pro-Latest DeepEP Local IPC Discriminator

**Date:** 2026-08-15

## 1. Test Script Information

- Test script:
  `/data/ycfeng/stepfun-performance-optimization/aiconfigurator-step4-pro/tests/performance/step4_pro_latest/run_step4_deepep_local_ipc_preflight.py`
- Host validation environment:
  - conda env:
    `/home/i-fengyicheng/miniconda3/envs/aic-step-design`
  - Python: `3.11.15`
  - Ruff: `0.14.1`
- B300 runtime:
  - image:
    `hub.stepfun-inc.com/stepcast/stepcast:2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled`
  - Python: `3.12.13`
  - vLLM: `0.19.0.post20.dev26+gc820e5ae1`
  - DeepEP: `1.2.1+c5284713.step.torch2.10.0.cu129.py312`
  - GPU: `8 x NVIDIA B300 SXM6 AC`, SM `10.3`
  - driver: `580.159.03`
- Static commands:

  ```bash
  CONDA_ENV=/home/i-fengyicheng/miniconda3/envs/aic-step-design
  "$CONDA_ENV/bin/python" -m py_compile \
    tests/performance/step4_pro_latest/run_step4_deepep_local_ipc_preflight.py
  "$CONDA_ENV/bin/ruff" check \
    tests/performance/step4_pro_latest/run_step4_deepep_local_ipc_preflight.py
  "$CONDA_ENV/bin/ruff" format --check \
    tests/performance/step4_pro_latest/run_step4_deepep_local_ipc_preflight.py
  git diff --check -- \
    tests/performance/step4_pro_latest/run_step4_deepep_local_ipc_preflight.py
  ```

- GPU predict-only command:

  ```bash
  sudo -n systemd-run --scope -p MemoryMax=3G --expand-environment=no \
    timeout 300s /kubebrain/rlaunch --predict-only \
    --charged-group=b300_train_infra \
    --private-machine=group \
    --positive-tags=B300 \
    --gpu=8 --cpu=32 --memory=200000 --backoff-limit=1 \
    -- bash -lc true
  ```

- Remote test command after creating the one-node holder and transferring the
  script:

  ```bash
  sudo -n systemd-run --scope -p MemoryMax=3G --expand-environment=no \
    timeout --signal=TERM --kill-after=30s 600s \
    /kubebrain/brainctl -n shai-core exec -i \
    replica/s4p-deepep-local-0815-2046-cfcd2084 -- \
    /bin/bash -lc '
      set -euo pipefail
      export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:/usr/local/nvidia/lib64
      export PYTHONUNBUFFERED=1
      nvidia-smi \
        --query-gpu=name,memory.used,memory.free,driver_version,compute_cap \
        --format=csv,noheader
      torchrun --standalone --nproc-per-node=8 \
        /home/step4-deepep-local-preflight/tests/performance/step4_pro_latest/run_step4_deepep_local_ipc_preflight.py
    ' </dev/null
  ```

## 2. Validation Criteria

1. World size is exactly `8`; all devices are B300 SM `10.3`.
2. Runtime vLLM version exactly equals
   `0.19.0.post20.dev26+gc820e5ae1`.
3. Every rank invokes the pinned vLLM NVSHMEM harmonization helper.
4. Every rank constructs `deep_ep.Buffer` with:
   - `num_nvl_bytes=1073741824`;
   - `num_rdma_bytes=0`;
   - `low_latency_mode=False`.
5. Construction acceptance requires `8/8`
   `STEP4_DEEPEP_LOCAL_IPC=PASS` markers and zero constructor
   `runtime.cu:83` errors.
6. Overall runtime acceptance additionally requires clean explicit
   `buffer.destroy()`, remote exit code `0`, and final RJob/Replica counts
   `0/0`.

## 3. Test Results and Evidence

**Overall outcome: FAIL.**

The construction discriminator passed, but explicit DeepEP teardown failed.

| Metric | Expected | Actual | Result |
|---|---:|---:|---|
| GPUs | 8 | 8 | PASS |
| Free memory per GPU before test | clean/available | `274114 MiB` | PASS |
| Buffer construction PASS markers | 8 | 8 | PASS |
| Constructor `runtime.cu:83` CUDA 999 | 0 | 0 | PASS |
| Explicit destroy `runtime.cu:29` CUDA 999 | 0 | 8 | FAIL |
| Remote exit code | 0 | 1 | FAIL |
| Final RJob count | 0 | 0 | PASS |
| Final Replica count | 0 | 0 | PASS |

Key error:

```text
RuntimeError: Failed: CUDA error
/workspace/DeepEP/csrc/kernels/runtime.cu:29 'unknown error'
```

Interpretation:

- Local eight-rank Buffer construction works with `num_rdma_bytes=0`.
- The earlier EP16 failure at `deep_ep.Buffer.runtime.sync` is therefore
  narrowed to the cross-node NVSHMEM/RDMA/shared-host-SHM path.
- Explicit local destroy has a separate CUDA `999` teardown defect.
- The available launcher lacks the required shared-host-SHM contract.
  EP16/EP32 DeepEP collection cannot proceed without platform/vendor action.

Evidence directory:

```text
/data/ycfeng/tmp/step4_aic_deepep_local_b300_20260815/s4p-deepep-local-0815-2046/
```

Evidence hashes:

```text
predict_only.log  7cb1596290d00c64f97700cc7af43160ae57e26d1c7f0c71895c20483f729a6c
launch.log        8e0051eec4ade9682df32ddbec5ee17b2661fa5120295bf820cd16a233fc3b38
remote_exec.log   5ad3bdfc06609d3f9a08f5de1866f755aed8c9c5fd8286e146fa049516b2fcd6
result.env        710eff5e3845eb076bb4d3993c458e8a8584480e2443078b361b0f63068bc867
cleanup_metrics.env ef6a0372d8ede15ae884b93260fb11dabdca62024bb86d03f88ad14a3a282e80
test script       46d020bdb7263e91533a702bfce682364ad9e97840684491b8b239a6f9300261
```
