## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-08-17 | Added a fail-fast quota-admission gate: `predict-only` proves per-worker node fit but not total replica quota, so a 2×8 B300 live run requires separate confirmation of at least 16 available B300 GPUs. |
| 2026-08-17 | Clarified the distributed AgRs evidence scope and required a coordinated two-node validation/shutdown barrier. |
| 2026-08-17 | Replaced the unavailable DeepEP runtime requirement with explicit vLLM `allgather_reducescatter` communication, clarified env/CLI selection semantics and runtime evidence limits, and retained NCCL `alltoall` only as a labeled AIC simulation proxy. |

# Step4-pro-v4 B300 性能建模需求

## 1. 要做什么

在 B300 上，用同一套模型配置和测试工况得到两组数据：

- vLLM 实测；
- 性能仿真器预测。

最后逐点对比时延、吞吐、显存和通信开销，说明误差来自哪里。

当前没有目标 checkpoint，也不需要等待。78 层目标模型使用 synthetic `config.json` 和
vLLM `--load-format dummy` 运行。真实权重以后只用于补充路由分布、生成质量和 MTP
接受率，不阻塞本次性能测试。

执行顺序：

1. 用仓库里的 14 层小模型脚本验证 Step4Pro 和 FA4 执行链路。
2. 按本文 Shape 生成 78 层 synthetic config。
3. 用 dummy weights 和非 DeepEP 通信在 B300 上跑 EP16、EP32 和 2×EP16。
4. 仿真器运行相同工况。
5. 输出实测与仿真的对比表和初步分析。

## 2. 固定输入

### 2.1 vLLM

```text
repo: git@gitlab.basemind.com:sys/stepcast/vllm.git
branch: xwx/step4pro-fa3-optimus
commit: 607d1641ee3fec43653fca510d717725828890c2
```

主要实现：

```text
vllm/model_executor/models/step4pro.py
vllm/v1/attention/backends/optimus_fa4.py
vllm/model_executor/layers/fused_moe/optimus_fp8_moe.py
```

仿真器的算子 shape、dtype、KV cache 和通信方式以这个 commit 的实际执行路径为准。

本次任务对通信后端有一项明确运行覆盖：由于当前 DeepEP/NVSHMEM
环境不能稳定启动，vLLM 实际运行不得自动选择 DeepEP，必须显式使用：

```text
VLLM_ALL2ALL_BACKEND=allgather_reducescatter
VLLM_ENABLE_SEQUENCE_PARALLEL=0
--all2all-backend allgather_reducescatter
```

`nccl` 不是该 vLLM 版本接受的 `all2all` backend 名称。上述 backend
由 `AgRsAll2AllManager` 实现：dispatch 使用 `all_gatherv`，combine 使用
`reduce_scatterv`，CUDA 通信路径使用 PyNccl/NCCL。它是 NCCL 支持的
MoE 通信替代方案，但不是字面意义上的直接 NCCL `alltoall`。

环境变量和 CLI 参数必须同时保留，但作用不同：

- `VLLM_ALL2ALL_BACKEND` 的“已设置”状态阻止 Step MoE 自动选择
  DeepEP；
- CLI 参数把同一 backend 写入 `parallel_config`，并使 DP=1 等不会进入
  Step MoE 默认处理的运行也保持自描述和一致。

`VLLM_ENABLE_SEQUENCE_PARALLEL=0` 是显式安全约束。pinned source 的 AgRs
分支本身不会自动开启 sequence parallelism，且当前脚本固定 TP=1；
显式设置用于拒绝外部覆盖并保证运行记录清楚。运行日志必须包含：

```text
Using AgRsAll2AllManager all2all manager
```

并且不得包含任何 `Using DeepEP*All2AllManager` 或
`Auto-configured ... VLLM_ALL2ALL_BACKEND=...` marker。不允许运行时
fallback 或自动 backend 改写。

该 AgRs manager marker 在 pinned vLLM 中通过
`logger.info_once(..., scope="global")` 输出，因此双节点验收要求整个
distributed job 至少出现一份，不要求每个 replica 各输出一份。每个
replica 必须分别证明：

- `allgather_reducescatter` 配置 marker 可见；
- 至少一个真实 batch forward 可见；
- DeepEP manager marker 和自动 backend 选择 marker 均为 `0`。

双节点 runner 必须在两个 replica 都完成上述本地验证后才开始退出：
host 先等待 `2/2` validation-ready，再通知两端进入 shutdown-armed；
确认 `2/2` armed 后，必须并发发出最终 shutdown。任何一个 replica
不得在另一端 armed 前独立停止 vLLM/TCPStore。

双节点 live run 还必须单独确认总 quota。当前 `brainctl rjob launch
--predict-only` 只返回单个 worker 可使用的节点资源：`--replica 2`
和诊断用的 `--replica 8` 都返回相同的 7 个单节点候选，因此它不能证明
总 quota 足以容纳 `2 × 8 = 16` 张 B300。提交 live run 前必须从平台
event、quota owner 或其他有权限的直接查询获得当前可用 B300 quota
`>=16` 的证据。若当前身份因 RBAC 不能读取 quota，且最近一次可信 event
仍显示小于 `16`，必须停止，不得用 predict-only 的节点列表代替 quota
证据。

`AgRsAll2AllManager` 当前没有与 DeepEP `backend=HT op=dispatch/combine`
等价的逐次调用日志。因此配置验证只能证明 backend/manager 选择、无自动
改写和真实 batch forward；在加入 AgRs profiler 或 provider marker 前，
不得声称已由日志逐次证明每个 dispatch/combine 调用。

### 2.2 模型 Shape

机器可读配置：
[step4pro_v4_shape_manifest.json](./step4pro_v4_shape_manifest.json)

```text
shape hash: 53103019932b93b20a60b6f9dfe6154be6330befdd6a6fa2f6cb67278fc03fde
```

| 参数 | 值 |
| --- | ---: |
| hidden size | 7168 |
| trunk layers | 78 |
| vocab size | 128896 |
| Dense FFN | layer 0、1，intermediate 26112 |
| Latent MoE | layer 2..77，共 76 层 |
| routed experts / topk | 896 / 16 |
| latent hidden / MoE intermediate | 3584 / 3584 |
| shared expert size | 3584 |
| MTP | layer 78，1 层，第一版启用 |
| 总参数量 | 约 2.657T |

Attention 配置：

| 类型 | 层数 | 主要参数 |
| --- | ---: | --- |
| Full MFA | 20 | Q heads=64，KV heads=1，head dim=512，Q LoRA=2048，no-PE/RoPE=448/64，O groups=8，O LoRA=1024 |
| SWA GQA | 58 | Q heads=128，KV heads=8，head dim=128，window=512 |

两种 Attention 都启用 head-wise gate。Full MFA 的逻辑 K/V 共用 storage。

当前使用 `S,S,S,F` 并将最后一层设为 Full：

```text
Full layer ids:
3,7,11,15,19,23,27,31,35,39,
43,47,51,55,59,63,67,71,75,77
```

这组顺序是当前 synthetic config 的固定输入。以后拿到 checkpoint 再核对，不影响本次测试。

### 2.3 精度和运行设置

```text
precision: fp8_experts_bf16_replicated
routed experts: FP8 E4M3FN，128×128 block，dynamic activation
attention / Dense / latent projection / shared expert / LM head: BF16
norm / router: FP32 path
KV cache: auto，当前预期解析为 BF16

tensor_parallel_size = 1
data_parallel_size = EP size
enable_expert_parallel = true
block_size = 128
```

FP8 scale 的逻辑 shape：

```text
w13 scale_inv: [E_local,56,28]
w2 scale_inv:  [E_local,28,28]
activation scale: [tokens or routed_rows,28]

E_local: EP16=56，EP32=28，EP64=14
```

KV cache 需要同时输出 requested/resolved dtype、logical bytes 和实际 allocated bytes。
Full/SWA 混合 allocator 可能产生 padding，不能只按逻辑 K/V 估算显存。

基础模型使用一层 MTP：SWA Attention、Dense FFN 和一个额外 LM head。本次只评估
MTP1，不要求构造或推算 MTP3/MTP5。当前分支复用了 Step3p5 MTP 基础设施，执行人需要
补齐 Step4Pro 的 MTP1 构造路径。

### 2.4 硬件和拓扑

```text
GPU: NVIDIA B300 SXM6 AC
Full MFA backend: Optimus CuTe FA4，block/page size 128
MoE compute backend: Optimus FP8 MoE / DeepGEMM
MoE communication backend: allgather_reducescatter / AgRs / NCCL
DeepEP status: disabled for active task runs
```

| 配置 | GPU | 组织方式 | 静态权重估算 |
| --- | ---: | --- | ---: |
| `ep16_r1` | 16 | 一个 EP16 实例 | 约 230 GB/GPU |
| `ep32_r1` | 32 | 一个 EP32 实例 | 约 148 GB/GPU |
| `ep16_r2` | 32 | 两个 EP16 实例 | 约 230 GB/GPU |

静态权重估算不含 scale、KV cache 和 workspace。资源允许时增加 `ep64_r1`。

## 3. 怎么跑

### 3.1 先跑小模型 smoke

目标分支已有两个脚本：

| 脚本 | 用途 |
| --- | --- |
| `rjob-step4pro-optimus-single.sh` | 14 层模型，1×B300，验证加载、hd512 FA4、prefill/decode 和并发请求 |
| `rjob-step4pro-2node.sh` | pinned source 中的历史 DeepEP 参考脚本；不再作为本任务的 active runtime 入口 |
| `tests/e2e/step4_pro_latest/run_b300_two_node_smoke.sh` | 当前 2×8 B300 active wrapper，验证 DP16+EP16、NCCL preflight 和 `AgRsAll2AllManager` |

先运行单卡脚本：

```bash
VLLM_PULL_COMMIT=607d1641ee3fec43653fca510d717725828890c2 \
bash rjob-step4pro-optimus-single.sh
```

脚本自带模型、tokenizer 和镜像默认值。需要确认服务健康、FA4 backend、prefill/decode 和
4 路并发请求。脚本在目标 commit 中通过了 `bash -n`，但仓库记录里的 B300 端到端验收
尚未完成，因此第一次运行仍按 smoke 处理。

### 3.2 再跑 78 层目标 Shape

准备一个只包含 `config.json` 的模型目录。配置必须包含第 2 节 Shape、78 项
`layer_types`、1M RoPE/最大长度设置，以及：

```json
{
  "model_type": "step4pro",
  "architectures": ["Step4ProForCausalLM"],
  "quantization_config": {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128]
  }
}
```

启动命令的公共部分：

```bash
VLLM_ALL2ALL_BACKEND=allgather_reducescatter \
VLLM_ENABLE_SEQUENCE_PARALLEL=0 \
vllm serve "${MODEL_CONFIG_DIR}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --served-model-name step4pro-v4-perf \
  --load-format dummy \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --kv-cache-dtype auto \
  --max-model-len 1048576 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.9
```

DP/EP 的跨节点资源、rank 和 RDMA 参数可参考 `rjob-step4pro-2node.sh`，
但不得沿用其 DeepEP/NVSHMEM backend 设置。加载 smoke 可以加
`--enforce-eager`；正式性能测试去掉它，并记录 CUDA Graph 是否生效。

Dummy weights 可以测算子时延、吞吐、显存、KV 和 AgRs 通信。它不能代表真实 expert 路由。
端到端运行记录实际 expert histogram，MoE microbenchmark 另外覆盖 balanced、轻度不均和
重度不均三种路由。

仿真侧现有 `--deepep-proxy b300_nccl_alltoall` 仍可作为显式、临时的
NCCL `alltoall` 通信 proxy，并必须继续标记为 `PROXY`。它与实际 vLLM
的 AgRs backend 不是同一个算法，因此在补齐 AgRs 实测/模型前，不得把
通信分项误差描述为同 backend 的精度误差，也不得把 proxy 当作真实 DeepEP
或真实 AgRs silicon 数据。

当前存在三个必须分开记录的通信身份：

1. active vLLM runtime：`allgather_reducescatter` / AgRs；
2. AIC exact operation identity：`vllm_deepep_high_throughput`，其真实
   B300 数据仍缺失；
3. 已完成仿真：显式 `b300_nccl_alltoall` `PROXY`。

AIC 尚无 AgRs-specific 通信模型。因此当前仿真结果不能作为 AgRs
same-backend 精度验证；补建或校准 AgRs 模型属于后续工作。

## 4. 测什么

### 4.1 Prefill

| Prompt tokens | Batch size |
| ---: | --- |
| 512 | 1、8、32 |
| 2,048 | 1、4、16 |
| 8,192 | 1、4、8 |
| 32,768 | 1、2、4 |
| 131,072 | 1、2 |
| 262,144 | 1、2，显存不足时保留 1 |
| 524,288 | 1 |
| 1,048,544 | 1 |

仿真器覆盖整张表。vLLM 至少完成全部 `batch=1` 点和每个长度下的最大可运行 batch。
32K、128K、1M 的 batch=1 还要扫描：

```text
max_num_batched_tokens: 8192, 32768, 65536
```

1M 请求使用 1,048,544 input + 1 output。记录 TTFT、prefill latency、input tok/s、chunk
数量、每个 chunk 的时间和峰值 HBM。无法运行时记录 OOM 和最大可运行长度，不缩短后
当作成功。

### 4.2 Decode

| Context tokens | Output tokens |
| ---: | ---: |
| 2,048 | 256 |
| 8,192 | 256 |
| 32,768 | 256 |
| 131,072 | 256 |
| 262,144 | 128 |
| 524,288 | 64 |
| 1,048,544 | 32 |

最后一行的 context + output 正好是 1,048,576。长上下文优先在 `ep32_r1` 上测试；
`ep16_r1` 仍做容量测试。

Decode 的目标不是固定 batch 报一个时延，而是在给定时延预算下寻找最大 Decode batch。
这里的 `B` 指稳定 Decode step 中实际参与计算的 active sequences，不是请求队列里的总并发。

TPS 指单请求的平均生成速度，不是实例的总吞吐。MTP1 接受率固定为 0.85：15% 的
iteration 输出 1 个 token，85% 输出 2 个 token，平均是 1.85 token/iteration。四档要求
换算如下，MTP1 两列已经计入接受率：

| 有效生成速度 | 有效 TPOT | MTP-off `T0(B)` 上限 | MTP1 最低 iteration 速率 | MTP1 `T1(B)` 上限 |
| ---: | ---: | ---: | ---: | ---: |
| 30 token/s | 33.33 ms/token | 33.33 ms | 16.22 iter/s | 61.67 ms |
| 50 token/s | 20.00 ms/token | 20.00 ms | 27.03 iter/s | 37.00 ms |
| 80 token/s | 12.50 ms/token | 12.50 ms | 43.24 iter/s | 23.13 ms |
| 100 token/s | 10.00 ms/token | 10.00 ms | 54.05 iter/s | 18.50 ms |

计算方式：

```text
effective_tps_mtp_off(B) = 1000 / T0(B)
effective_tps_mtp1(B) = 1.85 * 1000 / T1(B)
```

对每个 context、并行配置和 `S_tpot`，从 `B=1` 开始按
`1,2,4,8,16...` 增大，直到超过时延预算或 OOM；再在最后一个通过点和第一个失败点之间
补测，得到：

```text
B_max = 满足 effective_TPOT(B) <= S_tpot 的最大 B
```

四档要求都要给出 `B_max`。同时保留完整的 `B -> TPOT/吞吐/HBM` 数据，便于后续增加
其他时延档位。

Decode 时延必须单独测，不包含 prefill：

```text
first_decode_step_ms
steady_decode_step_ms: p50 / p90 / p99
ITL_ms: p50 / p90 / p99
TPOT_ms
decode_generation_ms: 首 token 到最后 token
end_to_end_ms: 包含 prefill
```

仿真器与 vLLM 主要对齐 `steady_decode_step_ms p50`。第一版用平均 TPOT 计算容量；
`ITL p99` 只用于观察服务抖动，不作为 `B_max` 的约束。每个 engine step 都要记录 active
sequences 和 batched tokens，避免把请求并发当成实际 Decode batch。

### 4.3 MTP1 时延换算

本次只比较两个配置：

| Variant | MTP layers | speculative tokens | 用途 |
| --- | ---: | ---: | --- |
| `mtp_off` | 0 | 0 | 普通 Decode 基线 |
| `mtp1` | 1 | 1 | 一次 draft 和一次 target verification |

MTP1 只测试以下代表 context：

```text
2K、128K、约 1M
```

每个 context 都按 4.2 搜索 batch。对每个 batch 记录不依赖接受率的原始成本：

```text
T0(B): MTP-off 的一次稳定 Decode step 时间
T1(B): MTP1 的一次完整 speculative iteration 时间
```

`T1` 包含一次 MTP forward、target verification、采样、KV 和调度。仿真器能拆分时可以
附加分项，但不是第一版的强制要求。

接受率不由 dummy weights 推断。本次固定假设 MTP1 draft 接受率 `A=0.85`，一次 MTP1
iteration 平均输出 1.85 个 token。每个 batch 的换算为：

```text
MTP-off effective_TPOT(B) = T0(B)
MTP1 effective_TPOT(B) = T1(B) / 1.85

MTP1 output_tok_s(B) = B * 1.85 / (T1(B) / 1000)
MTP1 speedup(B) = T0(B) * 1.85 / T1(B)

B_max_mtp1(S_tpot) = max B satisfying T1(B)/1.85 <= S_tpot
```

`1.85 = 1 + 0.85`，其中 1 是 target 给出的 token。`T0/T1` 标为实测或仿真原始值；
换算后的 TPOT、吞吐、speedup 和 `B_max` 标为 `A=0.85` 假设下的估算值。平均接受率
只能换算平均 TPOT 和吞吐，不能用于推导 ITL p90/p99。

### 4.4 必须做的对比

1. `ep16_r1` 对比 `ep32_r1`：相同时延预算下比较 `B_max` 和吞吐。
2. `ep32_r1` 对比 `ep16_r2`：相同 32 张 GPU 和时延预算，比较总承载量和吞吐。
3. MTP off 对比 MTP1：按 4.3 比较相同时延预算下的 `B_max` 和吞吐。
4. 长上下文：比较 128K、512K、1M 的 TTFT、decode step、KV 显存和最大并发。

## 5. 交付什么

每个工况至少重复三次。vLLM 和仿真器输出同一张结果表：

| 类别 | 指标 |
| --- | --- |
| 时延 | 时延预算、TTFT、prefill、first/steady decode step、ITL、TPOT、decode generation、E2E |
| 容量/吞吐 | `B_max`、首个失败 batch、input/output/total tok/s、tok/s/GPU |
| 显存 | 权重、scale、KV、workspace、峰值 HBM |
| 分项 | Full MFA、SWA、Dense、Latent MoE、MTP1 iteration、dispatch/combine |
| MoE | expert token histogram、max/mean load、padding ratio |
| 状态 | OOM、backend fallback、重试和异常日志 |

通信状态还必须记录：

- `runtime_all2all_backend=allgather_reducescatter`；
- 整个 distributed job 至少一份
  `runtime_all2all_manager=AgRsAll2AllManager`；
- `sequence_parallel=false`；
- 每个 replica 的 `backend_config_marker_count>=1`；
- 每个 replica 至少一份真实 batch forward；
- 所有 replica 合计 `agrs_manager_marker_count>=1`；
- DeepEP manager marker 数量为 `0`；
- backend 自动选择 marker 数量为 `0`；
- 双节点 validation-ready 和 shutdown-armed 均为 `2/2`；
- 仿真若使用 `b300_nccl_alltoall`，必须记录 `result_fidelity=PROXY`。

最终提交：

1. 78 层 synthetic `config.json` 和仿真器模型配置。
2. 小模型 smoke、EP16、EP32、2×EP16 的运行脚本和日志。
3. vLLM 原始数据、仿真结果和逐工况误差表。
4. 初步报告：主要瓶颈、相同时延预算下的 EP 扩展收益、EP32 与 2×EP16 的差别、1M
   容量边界、MTP1 的 `B_max` 和吞吐，以及仿真器下一步需要校准的参数。

Step4Pro MTP1 的 dummy-weight 构造仍需验证，不阻塞 MTP-off 的主体性能测试。

更细的实现考据见：
[vLLM B300 实测规格](./step4pro_v4_vllm_b300_8bit_task.md)。
