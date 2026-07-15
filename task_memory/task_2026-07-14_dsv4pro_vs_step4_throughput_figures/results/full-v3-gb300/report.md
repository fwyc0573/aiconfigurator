# Step4 vs DeepSeek-V4-Pro SOL Comparison

## Summary

| Metric | Value |
|---|---:|
| Mode runs | 16 |
| Normalized rows | 58 |
| Ranked rows | 58 |
| Paired comparisons | 8 |
| Unpaired comparisons | 0 |

## Modeling Boundary

Temporary MLA substitution is used for all 92 Step4 attention layers. The original labels remain 23 Full-MLA-approx and 69 SWA-MLA-approx; rows at ISL >= 65536 are approximation dominated.

All operation latency evidence in this comparison is required to use DatabaseMode.SOL. Primary OSL=1 rows rank by fixed-cluster Prefill input throughput; the 4K/1024 decode smoke ranks by fixed-cluster output throughput.

## Rank-One Results

| Model | System | Workload | ISL | OSL | TTFT SLA (ms) | Mode | Metric | Value | TTFT (ms) | Config |
|---|---|---|---:|---:|---:|---|---|---:|---:|---|
| deepseek-ai/DeepSeek-V4-Pro | gb300 | throughput | 1024 | 1024 | 5000 | disagg | output_token_throughput | 12741.86752 | 71.207 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=2|p_workers=1|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1803|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | throughput | 131072 | 1024 | 5000 | disagg | output_token_throughput | 19.1808 | 3002.28 | disagg_AA|p_tp=8|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=32|p_cp=1|p_bs=1|p_workers=1|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=2 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | throughput | 16384 | 1024 | 5000 | disagg | output_token_throughput | 1150.3296 | 701.031 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=336|d_workers=2 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | throughput | 2048 | 1024 | 5000 | disagg | output_token_throughput | 8664.336 | 265.917 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=4|p_workers=5|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1570|d_workers=3 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | throughput | 32768 | 1024 | 5000 | disagg | output_token_throughput | 448.632 | 481.543 | disagg_AA|p_tp=4|p_pp=1|p_dp=1|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=1|p_workers=15|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=252|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | throughput | 4096 | 1024 | 5000 | disagg | output_token_throughput | 5059.2384 | 273.243 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=2|p_workers=3|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1235|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | throughput | 65536 | 1024 | 5000 | disagg | output_token_throughput | 157.65120000000002 | 1278.957 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=40|d_workers=2 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | throughput | 8192 | 1024 | 5000 | disagg | output_token_throughput | 2663.2943999999998 | 281.146 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=1|p_workers=13|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=633|d_workers=3 |
| stepfun-ai/Step4 | gb300 | throughput | 1024 | 1024 | 5000 | disagg | output_token_throughput | 14944.848000000002 | 62.893 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=3|p_workers=6|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1878|d_workers=5 |
| stepfun-ai/Step4 | gb300 | throughput | 131072 | 1024 | 5000 | disagg | output_token_throughput | 24.0912 | 4176.999 | disagg_AA|p_tp=8|p_pp=1|p_dp=1|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=2 |
| stepfun-ai/Step4 | gb300 | throughput | 16384 | 1024 | 5000 | disagg | output_token_throughput | 1008.8063999999999 | 1370.395 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=2|p_workers=6|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=119|d_workers=1 |
| stepfun-ai/Step4 | gb300 | throughput | 2048 | 1024 | 5000 | disagg | output_token_throughput | 9636.5952 | 84.035 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=2|p_workers=2|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1253|d_workers=1 |
| stepfun-ai/Step4 | gb300 | throughput | 32768 | 1024 | 5000 | disagg | output_token_throughput | 343.5648 | 2179.636 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=1|p_workers=14|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=114|d_workers=1 |
| stepfun-ai/Step4 | gb300 | throughput | 4096 | 1024 | 5000 | disagg | output_token_throughput | 5206.81728 | 96.679 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=1|p_workers=10|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=752|d_workers=3 |
| stepfun-ai/Step4 | gb300 | throughput | 65536 | 1024 | 5000 | disagg | output_token_throughput | 99.9936 | 4034.494 | disagg_AA|p_tp=2|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=30|d_workers=1 |
| stepfun-ai/Step4 | gb300 | throughput | 8192 | 1024 | 5000 | disagg | output_token_throughput | 2385.936 | 2897.061 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=12|p_workers=10|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=168|d_workers=3 |

## Paired Model Deltas

Absolute delta is Step4 minus DeepSeek-V4-Pro. TPOT is observational and does not affect eligibility.

| Metric | Step4 | DeepSeek-V4-Pro | Absolute delta | Relative delta | Polarity | Status |
|---|---:|---:|---:|---:|---|---|
| ranking_metric_value | 14944.848000000002 | 12741.86752 | 2202.980480000002 | 0.17289306112641187 | higher_is_better | computed |
| ttft | 62.893 | 71.207 | -8.313999999999993 | -0.11675818388641557 | lower_is_better | computed |
| tpot | 72.326 | 65.155 | 7.170999999999992 | 0.11006062466426203 | lower_is_better | computed |
| request_latency | 74052.39099999999 | 66724.772 | 7327.6189999999915 | 0.10981856933134207 | lower_is_better | computed |
| ranking_metric_value | 24.0912 | 19.1808 | 4.910399999999999 | 0.2560060060060059 | higher_is_better | computed |
| ttft | 4176.999 | 3002.28 | 1174.7189999999996 | 0.39127563052080405 | lower_is_better | computed |
| tpot | 4.292 | 4.783 | -0.49100000000000055 | -0.10265523729876658 | lower_is_better | computed |
| request_latency | 8567.715 | 7895.289000000001 | 672.4259999999995 | 0.08516800334984564 | lower_is_better | computed |
| ranking_metric_value | 1008.8063999999999 | 1150.3296 | -141.5232000000001 | -0.1230283911671925 | higher_is_better | computed |
| ttft | 1370.395 | 701.031 | 669.364 | 0.9548279605324159 | lower_is_better | computed |
| tpot | 26.479 | 32.821 | -6.341999999999999 | -0.19322994424301512 | lower_is_better | computed |
| request_latency | 28458.412 | 34276.914 | -5818.501999999997 | -0.1697498788834957 | lower_is_better | computed |
| ranking_metric_value | 9636.5952 | 8664.336 | 972.2592000000004 | 0.11221393076168797 | higher_is_better | computed |
| ttft | 84.035 | 265.917 | -181.88199999999998 | -0.6839803397300661 | lower_is_better | computed |
| tpot | 59.87 | 61.215 | -1.345000000000006 | -0.021971738952871123 | lower_is_better | computed |
| request_latency | 61331.045 | 62888.86200000001 | -1557.81700000001 | -0.024770952287227106 | lower_is_better | computed |
| ranking_metric_value | 343.5648 | 448.632 | -105.06720000000001 | -0.2341946183063179 | higher_is_better | computed |
| ttft | 2179.636 | 481.543 | 1698.0929999999998 | 3.526357978415219 | lower_is_better | computed |
| tpot | 38.196 | 31.522 | 6.6739999999999995 | 0.21172514434363301 | lower_is_better | computed |
| request_latency | 41254.14399999999 | 32728.549 | 8525.594999999994 | 0.26049413311906966 | lower_is_better | computed |
| ranking_metric_value | 5206.81728 | 5059.2384 | 147.57888000000003 | 0.02917017707645483 | higher_is_better | computed |
| ttft | 96.679 | 273.243 | -176.564 | -0.6461794080726679 | lower_is_better | computed |
| tpot | 49.876 | 54.964 | -5.088000000000001 | -0.0925696819736555 | lower_is_better | computed |
| request_latency | 51119.827 | 56501.415 | -5381.588000000003 | -0.09524695974428257 | lower_is_better | computed |
| ranking_metric_value | 99.9936 | 157.65120000000002 | -57.657600000000016 | -0.36572890025575455 | higher_is_better | computed |
| ttft | 4034.494 | 1278.957 | 2755.5370000000003 | 2.154518877491581 | lower_is_better | computed |
| tpot | 33.525 | 28.175 | 5.349999999999998 | 0.18988464951197861 | lower_is_better | computed |
| request_latency | 38330.568999999996 | 30101.982 | 8228.586999999996 | 0.2733569836032722 | lower_is_better | computed |
| ranking_metric_value | 2385.936 | 2663.2943999999998 | -277.3583999999996 | -0.10414109683105241 | higher_is_better | computed |
| ttft | 2897.061 | 281.146 | 2615.915 | 9.304471697978986 | lower_is_better | computed |
| tpot | 23.785 | 40.128 | -16.343 | -0.4072717304625199 | lower_is_better | computed |
| request_latency | 27229.116 | 41332.090000000004 | -14102.974000000002 | -0.34121124772543565 | lower_is_better | computed |
