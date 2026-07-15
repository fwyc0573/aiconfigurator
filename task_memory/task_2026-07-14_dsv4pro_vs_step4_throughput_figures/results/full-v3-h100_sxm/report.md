# Step4 vs DeepSeek-V4-Pro SOL Comparison

## Summary

| Metric | Value |
|---|---:|
| Mode runs | 16 |
| Normalized rows | 27 |
| Ranked rows | 27 |
| Paired comparisons | 5 |
| Unpaired comparisons | 1 |

## Modeling Boundary

Temporary MLA substitution is used for all 92 Step4 attention layers. The original labels remain 23 Full-MLA-approx and 69 SWA-MLA-approx; rows at ISL >= 65536 are approximation dominated.

All operation latency evidence in this comparison is required to use DatabaseMode.SOL. Primary OSL=1 rows rank by fixed-cluster Prefill input throughput; the 4K/1024 decode smoke ranks by fixed-cluster output throughput.

## Rank-One Results

| Model | System | Workload | ISL | OSL | TTFT SLA (ms) | Mode | Metric | Value | TTFT (ms) | Config |
|---|---|---|---:|---:|---:|---|---|---:|---:|---|
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | throughput | 1024 | 1024 | 5000 | disagg | output_token_throughput | 592.71552 | 388.36 | disagg_AA|p_tp=2|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=1|p_workers=1|d_tp=2|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=436|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | throughput | 16384 | 1024 | 5000 | disagg | output_token_throughput | 15.494400000000002 | 3718.155 | disagg_AA|p_tp=8|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=32|p_cp=1|p_bs=1|p_workers=1|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | throughput | 2048 | 1024 | 5000 | disagg | output_token_throughput | 296.3232 | 777.547 | disagg_AA|p_tp=2|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=1|p_workers=1|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=23|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | throughput | 4096 | 1024 | 5000 | disagg | output_token_throughput | 184.07520000000002 | 1877.68 | disagg_AA|p_tp=4|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=2|p_workers=3|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=33|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | throughput | 8192 | 1024 | 5000 | disagg | output_token_throughput | 91.71360000000001 | 1884.238 | disagg_AA|p_tp=4|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=1|p_workers=3|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=12|d_workers=1 |
| stepfun-ai/Step4 | h100_sxm | throughput | 1024 | 1024 | 5000 | disagg | output_token_throughput | 1296.58176 | 88.772 | disagg_BA|p_tp=8|p_pp=1|p_dp=1|p_moe_tp=8|p_moe_ep=1|p_cp=1|p_bs=4|p_workers=2|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=265|d_workers=3 |
| stepfun-ai/Step4 | h100_sxm | throughput | 16384 | 1024 | 5000 | disagg | output_token_throughput | 191.02144 | 646.304 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=5|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=31|d_workers=1 |
| stepfun-ai/Step4 | h100_sxm | throughput | 2048 | 1024 | 5000 | disagg | output_token_throughput | 1090.4832 | 422.566 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=8|p_workers=2|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=119|d_workers=3 |
| stepfun-ai/Step4 | h100_sxm | throughput | 32768 | 1024 | 5000 | disagg | output_token_throughput | 24.062400000000004 | 3590.948 | disagg_AA|p_tp=8|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=1|p_workers=3|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h100_sxm | throughput | 4096 | 1024 | 5000 | disagg | output_token_throughput | 690.72128 | 124.503 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=2|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=106|d_workers=1 |
| stepfun-ai/Step4 | h100_sxm | throughput | 8192 | 1024 | 5000 | disagg | output_token_throughput | 444.384 | 518.454 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=2|p_workers=2|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=36|d_workers=1 |

## Paired Model Deltas

Absolute delta is Step4 minus DeepSeek-V4-Pro. TPOT is observational and does not affect eligibility.

| Metric | Step4 | DeepSeek-V4-Pro | Absolute delta | Relative delta | Polarity | Status |
|---|---:|---:|---:|---:|---|---|
| ranking_metric_value | 1296.58176 | 592.71552 | 703.8662400000001 | 1.1875279392042917 | higher_is_better | computed |
| ttft | 88.772 | 388.36 | -299.588 | -0.771418271706664 | lower_is_better | computed |
| tpot | 141.162 | 169.35 | -28.187999999999988 | -0.1664481842338352 | lower_is_better | computed |
| request_latency | 144497.498 | 173633.40999999997 | -29135.911999999982 | -0.16780130045248773 | lower_is_better | computed |
| ranking_metric_value | 191.02144 | 15.494400000000002 | 175.52704 | 11.328418009087152 | higher_is_better | computed |
| ttft | 646.304 | 3718.155 | -3071.851 | -0.826176154571286 | lower_is_better | computed |
| tpot | 37.361 | 11.912 | 25.448999999999998 | 2.136417058428475 | lower_is_better | computed |
| request_latency | 38866.606999999996 | 15904.131000000001 | 22962.475999999995 | 1.4438057634208366 | lower_is_better | computed |
| ranking_metric_value | 1090.4832 | 296.3232 | 794.1599999999999 | 2.6800466517640196 | higher_is_better | computed |
| ttft | 422.566 | 777.547 | -354.98100000000005 | -0.4565396046798458 | lower_is_better | computed |
| tpot | 73.625 | 34.664 | 38.961 | 1.1239614585737363 | lower_is_better | computed |
| request_latency | 75740.941 | 36238.819 | 39502.122 | 1.0900499268477817 | lower_is_better | computed |
| ranking_metric_value | 690.72128 | 184.07520000000002 | 506.64608 | 2.7523864159865092 | higher_is_better | computed |
| ttft | 124.503 | 1877.68 | -1753.1770000000001 | -0.9336931745558349 | lower_is_better | computed |
| tpot | 70.663 | 40.157 | 30.506 | 0.7596683019149838 | lower_is_better | computed |
| request_latency | 72412.752 | 42958.291 | 29454.460999999996 | 0.6856525321270345 | lower_is_better | computed |
| ranking_metric_value | 444.384 | 91.71360000000001 | 352.6704 | 3.845344638090751 | higher_is_better | computed |
| ttft | 518.454 | 1884.238 | -1365.784 | -0.7248468611714657 | lower_is_better | computed |
| tpot | 36.29 | 28.733 | 7.556999999999999 | 0.2630076915045418 | lower_is_better | computed |
| request_latency | 37643.123999999996 | 31278.097 | 6365.026999999995 | 0.20349789822571349 | lower_is_better | computed |
