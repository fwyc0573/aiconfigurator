# Step4 vs DeepSeek-V4-Pro SOL Comparison

## Summary

| Metric | Value |
|---|---:|
| Mode runs | 16 |
| Normalized rows | 44 |
| Ranked rows | 44 |
| Paired comparisons | 5 |
| Unpaired comparisons | 2 |

## Modeling Boundary

Temporary MLA substitution is used for all 92 Step4 attention layers. The original labels remain 23 Full-MLA-approx and 69 SWA-MLA-approx; rows at ISL >= 65536 are approximation dominated.

All operation latency evidence in this comparison is required to use DatabaseMode.SOL. Primary OSL=1 rows rank by fixed-cluster Prefill input throughput; the 4K/1024 decode smoke ranks by fixed-cluster output throughput.

## Rank-One Results

| Model | System | Workload | ISL | OSL | TTFT SLA (ms) | Mode | Metric | Value | TTFT (ms) | Config |
|---|---|---|---:|---:|---:|---|---|---:|---:|---|
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | throughput | 1024 | 1024 | 5000 | disagg | output_token_throughput | 3932.9856000000004 | 234.327 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=2|p_workers=1|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=417|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | throughput | 16384 | 1024 | 5000 | disagg | output_token_throughput | 224.4816 | 449.122 | disagg_AA|p_tp=8|p_pp=1|p_dp=1|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=56|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | throughput | 2048 | 1024 | 5000 | disagg | output_token_throughput | 2433.0240000000003 | 236.741 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=5|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=287|d_workers=3 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | throughput | 4096 | 1024 | 5000 | disagg | output_token_throughput | 1200.0096 | 288.001 | disagg_AA|p_tp=2|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=3|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=178|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | throughput | 8192 | 1024 | 5000 | disagg | output_token_throughput | 559.6415999999999 | 360.25 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=162|d_workers=1 |
| stepfun-ai/Step4 | h200_sxm | throughput | 1024 | 1024 | 5000 | disagg | output_token_throughput | 5977.584000000001 | 1792.296 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=31|p_workers=3|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=514|d_workers=5 |
| stepfun-ai/Step4 | h200_sxm | throughput | 16384 | 1024 | 5000 | disagg | output_token_throughput | 366.85440000000006 | 1884.038 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=3|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=50|d_workers=1 |
| stepfun-ai/Step4 | h200_sxm | throughput | 2048 | 1024 | 5000 | disagg | output_token_throughput | 3727.4687999999996 | 1854.346 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=15|p_workers=1|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=342|d_workers=1 |
| stepfun-ai/Step4 | h200_sxm | throughput | 32768 | 1024 | 5000 | disagg | output_token_throughput | 109.55520000000001 | 3154.329 | disagg_AA|p_tp=2|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=3|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=12|d_workers=1 |
| stepfun-ai/Step4 | h200_sxm | throughput | 4096 | 1024 | 5000 | disagg | output_token_throughput | 1971.78816 | 279.399 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=5|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=213|d_workers=3 |
| stepfun-ai/Step4 | h200_sxm | throughput | 65536 | 1024 | 5000 | disagg | output_token_throughput | 33.3648 | 3021.792 | disagg_AA|p_tp=8|p_pp=1|p_dp=1|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=8|d_workers=1 |
| stepfun-ai/Step4 | h200_sxm | throughput | 8192 | 1024 | 5000 | disagg | output_token_throughput | 839.376 | 3431.264 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=5|p_workers=5|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=70|d_workers=3 |

## Paired Model Deltas

Absolute delta is Step4 minus DeepSeek-V4-Pro. TPOT is observational and does not affect eligibility.

| Metric | Step4 | DeepSeek-V4-Pro | Absolute delta | Relative delta | Polarity | Status |
|---|---:|---:|---:|---:|---|---|
| ranking_metric_value | 5977.584000000001 | 3932.9856000000004 | 2044.5984000000003 | 0.519859111612308 | higher_is_better | computed |
| ttft | 1792.296 | 234.327 | 1557.969 | 6.648696052951645 | lower_is_better | computed |
| tpot | 48.4 | 47.746 | 0.6539999999999964 | 0.013697482511623933 | lower_is_better | computed |
| request_latency | 51305.496 | 49078.485 | 2227.0109999999986 | 0.0453765229305672 | lower_is_better | computed |
| ranking_metric_value | 366.85440000000006 | 224.4816 | 142.37280000000007 | 0.6342292642247742 | higher_is_better | computed |
| ttft | 1884.038 | 449.122 | 1434.916 | 3.19493589715044 | lower_is_better | computed |
| tpot | 30.596 | 27.943 | 2.6529999999999987 | 0.09494327738610737 | lower_is_better | computed |
| request_latency | 33183.746 | 29034.811 | 4148.934999999998 | 0.14289519570146322 | lower_is_better | computed |
| ranking_metric_value | 3727.4687999999996 | 2433.0240000000003 | 1294.4447999999993 | 0.5320312499999996 | higher_is_better | computed |
| ttft | 1854.346 | 236.741 | 1617.605 | 6.8328046261526305 | lower_is_better | computed |
| tpot | 41.314 | 39.807 | 1.506999999999998 | 0.037857663225060866 | lower_is_better | computed |
| request_latency | 44118.568 | 40959.302 | 3159.265999999996 | 0.0771318319828789 | lower_is_better | computed |
| ranking_metric_value | 1971.78816 | 1200.0096 | 771.77856 | 0.6431436548507611 | higher_is_better | computed |
| ttft | 279.399 | 288.001 | -8.601999999999975 | -0.029867951847389336 | lower_is_better | computed |
| tpot | 37.305 | 33.362 | 3.942999999999998 | 0.11818835801210952 | lower_is_better | computed |
| request_latency | 38442.414 | 34417.327 | 4025.0869999999995 | 0.11694943654398263 | lower_is_better | computed |
| ranking_metric_value | 839.376 | 559.6415999999999 | 279.73440000000005 | 0.4998456154796214 | higher_is_better | computed |
| ttft | 3431.264 | 360.25 | 3071.014 | 8.524674531575295 | lower_is_better | computed |
| tpot | 28.151 | 32.497 | -4.346 | -0.13373542173123673 | lower_is_better | computed |
| request_latency | 32229.736999999997 | 33604.681 | -1374.9439999999995 | -0.040915252253101275 | lower_is_better | computed |
