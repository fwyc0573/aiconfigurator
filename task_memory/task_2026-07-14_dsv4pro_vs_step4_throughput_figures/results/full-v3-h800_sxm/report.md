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
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | throughput | 1024 | 1024 | 5000 | disagg | output_token_throughput | 592.74496 | 388.339 | disagg_AA|p_tp=2|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=1|p_workers=1|d_tp=2|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=436|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | throughput | 16384 | 1024 | 5000 | disagg | output_token_throughput | 15.494400000000002 | 3718.079 | disagg_AA|p_tp=8|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=32|p_cp=1|p_bs=1|p_workers=1|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | throughput | 2048 | 1024 | 5000 | disagg | output_token_throughput | 296.3232 | 777.504 | disagg_AA|p_tp=2|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=1|p_workers=1|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=23|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | throughput | 4096 | 1024 | 5000 | disagg | output_token_throughput | 184.07520000000002 | 1877.585 | disagg_AA|p_tp=4|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=2|p_workers=3|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=33|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | throughput | 8192 | 1024 | 5000 | disagg | output_token_throughput | 91.71360000000001 | 1884.14 | disagg_AA|p_tp=4|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=1|p_workers=3|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=12|d_workers=1 |
| stepfun-ai/Step4 | h800_sxm | throughput | 1024 | 1024 | 5000 | disagg | output_token_throughput | 1269.9072 | 181.433 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=4|p_workers=2|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=265|d_workers=3 |
| stepfun-ai/Step4 | h800_sxm | throughput | 16384 | 1024 | 5000 | disagg | output_token_throughput | 187.488 | 921.816 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=6|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=31|d_workers=1 |
| stepfun-ai/Step4 | h800_sxm | throughput | 2048 | 1024 | 5000 | disagg | output_token_throughput | 797.76512 | 103.689 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=2|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=177|d_workers=1 |
| stepfun-ai/Step4 | h800_sxm | throughput | 32768 | 1024 | 5000 | disagg | output_token_throughput | 22.5504 | 3832.023 | disagg_AA|p_tp=8|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=1|p_workers=3|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h800_sxm | throughput | 4096 | 1024 | 5000 | disagg | output_token_throughput | 631.1232000000001 | 730.137 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=4|p_workers=2|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=80|d_workers=1 |
| stepfun-ai/Step4 | h800_sxm | throughput | 8192 | 1024 | 5000 | disagg | output_token_throughput | 290.1888 | 794.03 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=2|p_workers=2|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=16|d_workers=1 |

## Paired Model Deltas

Absolute delta is Step4 minus DeepSeek-V4-Pro. TPOT is observational and does not affect eligibility.

| Metric | Step4 | DeepSeek-V4-Pro | Absolute delta | Relative delta | Polarity | Status |
|---|---:|---:|---:|---:|---|---|
| ranking_metric_value | 1269.9072 | 592.74496 | 677.1622400000001 | 1.1424175416017035 | higher_is_better | computed |
| ttft | 181.433 | 388.339 | -206.906 | -0.5327973754889413 | lower_is_better | computed |
| tpot | 141.162 | 169.346 | -28.183999999999997 | -0.16642849550624164 | lower_is_better | computed |
| request_latency | 144590.15899999999 | 173629.29700000002 | -29039.138000000035 | -0.16724791554042884 | lower_is_better | computed |
| ranking_metric_value | 187.488 | 15.494400000000002 | 171.9936 | 11.100371747211893 | higher_is_better | computed |
| ttft | 921.816 | 3718.079 | -2796.263 | -0.7520719704987441 | lower_is_better | computed |
| tpot | 37.361 | 11.912 | 25.448999999999998 | 2.136417058428475 | lower_is_better | computed |
| request_latency | 39142.119 | 15904.055 | 23238.064 | 1.4611408222619953 | lower_is_better | computed |
| ranking_metric_value | 797.76512 | 296.3232 | 501.44192000000004 | 1.6922128270753019 | higher_is_better | computed |
| ttft | 103.689 | 777.504 | -673.815 | -0.8666386282257069 | lower_is_better | computed |
| tpot | 102.16 | 34.664 | 67.496 | 1.9471497807523654 | lower_is_better | computed |
| request_latency | 104613.36899999999 | 36238.776000000005 | 68374.593 | 1.8867798680617685 | lower_is_better | computed |
| ranking_metric_value | 631.1232000000001 | 184.07520000000002 | 447.0480000000001 | 2.4286161307987175 | higher_is_better | computed |
| ttft | 730.137 | 1877.585 | -1147.448 | -0.6111297224892616 | lower_is_better | computed |
| tpot | 57.03 | 40.157 | 16.873000000000005 | 0.42017580994596226 | lower_is_better | computed |
| request_latency | 59071.827000000005 | 42958.195999999996 | 16113.631000000008 | 0.37510027190154843 | lower_is_better | computed |
| ranking_metric_value | 290.1888 | 91.71360000000001 | 198.4752 | 2.164075993091537 | higher_is_better | computed |
| ttft | 794.03 | 1884.14 | -1090.1100000000001 | -0.5785716560340527 | lower_is_better | computed |
| tpot | 24.506 | 28.733 | -4.227 | -0.1471130755577211 | lower_is_better | computed |
| request_latency | 25863.667999999998 | 31277.999 | -5414.331000000002 | -0.17310349680617362 | lower_is_better | computed |
