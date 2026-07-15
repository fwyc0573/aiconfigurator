# Step4 vs DeepSeek-V4-Pro SOL Comparison

## Summary

| Metric | Value |
|---|---:|
| Mode runs | 480 |
| Normalized rows | 545 |
| Ranked rows | 545 |
| Paired comparisons | 89 |
| Unpaired comparisons | 28 |

## Modeling Boundary

Temporary MLA substitution is used for all 92 Step4 attention layers. The original labels remain 23 Full-MLA-approx and 69 SWA-MLA-approx; rows at ISL >= 65536 are approximation dominated.

All operation latency evidence in this comparison is required to use DatabaseMode.SOL. Primary OSL=1 rows rank by fixed-cluster Prefill input throughput; the 4K/1024 decode smoke ranks by fixed-cluster output throughput.

## Rank-One Results

| Model | System | Workload | ISL | OSL | TTFT SLA (ms) | Mode | Metric | Value | TTFT (ms) | Config |
|---|---|---|---:|---:|---:|---|---|---:|---:|---|
| deepseek-ai/DeepSeek-V4-Pro | gb300 | decode_smoke | 4096 | 1024 | 1000 | agg | output_token_throughput | 5686.8983915650015 | 421.9725282053259 | agg_patternA|tp=1|pp=1|dp=4|moe_tp=1|moe_ep=4|cp=1|bs=727|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | decode_smoke | 4096 | 1024 | 1000 | disagg | output_token_throughput | 5059.2384 | 273.243 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=2|p_workers=3|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1235|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | decode_smoke | 4096 | 1024 | 200 | agg | output_token_throughput | 940.3004636798224 | 199.89558104822441 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=158|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | decode_smoke | 4096 | 1024 | 200 | disagg | output_token_throughput | 5054.2272 | 136.758 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=1|p_workers=6|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1021|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | decode_smoke | 4096 | 1024 | 2000 | agg | output_token_throughput | 5686.8983915650015 | 421.9725282053259 | agg_patternA|tp=1|pp=1|dp=4|moe_tp=1|moe_ep=4|cp=1|bs=727|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | decode_smoke | 4096 | 1024 | 2000 | disagg | output_token_throughput | 5059.2384 | 273.243 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=2|p_workers=3|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1235|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | decode_smoke | 4096 | 1024 | 500 | agg | output_token_throughput | 5686.8983915650015 | 421.9725282053259 | agg_patternA|tp=1|pp=1|dp=4|moe_tp=1|moe_ep=4|cp=1|bs=727|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | decode_smoke | 4096 | 1024 | 500 | disagg | output_token_throughput | 5059.2384 | 273.243 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=2|p_workers=3|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1235|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | decode_smoke | 4096 | 1024 | 5000 | agg | output_token_throughput | 5686.8983915650015 | 421.9725282053259 | agg_patternA|tp=1|pp=1|dp=4|moe_tp=1|moe_ep=4|cp=1|bs=727|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | decode_smoke | 4096 | 1024 | 5000 | disagg | output_token_throughput | 5059.2384 | 273.243 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=2|p_workers=3|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1235|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 16384 | 1 | 1000 | agg | prefill_input_throughput | 8914346.540325223 | 917.1301522794208 | agg_patternA|tp=4|pp=1|dp=4|moe_tp=1|moe_ep=16|cp=1|bs=1996|ctx_tokens=16384 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 16384 | 1 | 1000 | disagg | prefill_input_throughput | 20449.880247806446 | 701.031 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=2 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 16384 | 1 | 200 | disagg | prefill_input_throughput | 9215.736693237337 | 194.45 | disagg_AA|p_tp=8|p_pp=1|p_dp=1|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=2 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 16384 | 1 | 2000 | agg | prefill_input_throughput | 9718746.339774225 | 1376.4672450861365 | agg_patternA|tp=2|pp=1|dp=8|moe_tp=1|moe_ep=16|cp=1|bs=1633|ctx_tokens=16384 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 16384 | 1 | 2000 | disagg | prefill_input_throughput | 20449.880247806446 | 701.031 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=2 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 16384 | 1 | 500 | agg | prefill_input_throughput | 1982473.3251880198 | 499.9976480924355 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=484|ctx_tokens=16384 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 16384 | 1 | 500 | disagg | prefill_input_throughput | 20312.57025425096 | 378.091 | disagg_AA|p_tp=2|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=1|p_workers=15|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 16384 | 1 | 5000 | agg | prefill_input_throughput | 9718746.339774225 | 1376.4672450861365 | agg_patternA|tp=2|pp=1|dp=8|moe_tp=1|moe_ep=16|cp=1|bs=1633|ctx_tokens=16384 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 16384 | 1 | 5000 | disagg | prefill_input_throughput | 20449.880247806446 | 701.031 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=2 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 4096 | 1 | 1000 | agg | prefill_input_throughput | 17173713.065674055 | 677.828490291194 | agg_patternA|tp=1|pp=1|dp=8|moe_tp=1|moe_ep=8|cp=1|bs=2842|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 4096 | 1 | 1000 | disagg | prefill_input_throughput | 28106.84994675069 | 273.243 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=2|p_workers=15|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 4096 | 1 | 200 | agg | prefill_input_throughput | 404714.7927922588 | 199.88397123285813 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=158|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 4096 | 1 | 200 | disagg | prefill_input_throughput | 28078.79612161628 | 136.758 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=1|p_workers=15|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 4096 | 1 | 2000 | agg | prefill_input_throughput | 17173713.065674055 | 677.828490291194 | agg_patternA|tp=1|pp=1|dp=8|moe_tp=1|moe_ep=8|cp=1|bs=2842|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 4096 | 1 | 2000 | disagg | prefill_input_throughput | 28106.84994675069 | 273.243 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=2|p_workers=15|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 4096 | 1 | 500 | agg | prefill_input_throughput | 7648177.920087521 | 426.8352585556277 | agg_patternA|tp=1|pp=1|dp=4|moe_tp=1|moe_ep=4|cp=1|bs=797|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 4096 | 1 | 500 | disagg | prefill_input_throughput | 28106.84994675069 | 273.243 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=2|p_workers=15|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 4096 | 1 | 5000 | agg | prefill_input_throughput | 17173713.065674055 | 677.828490291194 | agg_patternA|tp=1|pp=1|dp=8|moe_tp=1|moe_ep=8|cp=1|bs=2842|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 4096 | 1 | 5000 | disagg | prefill_input_throughput | 28106.84994675069 | 273.243 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=2|p_workers=15|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 65536 | 1 | 1000 | agg | prefill_input_throughput | 9084.268864609054 | 901.7786815970189 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=1|ctx_tokens=65536 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 65536 | 1 | 1000 | disagg | prefill_input_throughput | 8403.489417676168 | 852.979 | disagg_AA|p_tp=8|p_pp=1|p_dp=1|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=2 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 65536 | 1 | 2000 | agg | prefill_input_throughput | 946034.6902756813 | 1775.1568914567206 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=205|ctx_tokens=65536 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 65536 | 1 | 2000 | disagg | prefill_input_throughput | 11209.133692532274 | 1278.957 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=2 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 65536 | 1 | 5000 | agg | prefill_input_throughput | 1259391.3052697089 | 2328.693224836844 | agg_patternA|tp=8|pp=1|dp=2|moe_tp=1|moe_ep=16|cp=1|bs=358|ctx_tokens=65536 |
| deepseek-ai/DeepSeek-V4-Pro | gb300 | primary | 65536 | 1 | 5000 | disagg | prefill_input_throughput | 11209.133692532274 | 1278.957 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=2 |
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | decode_smoke | 4096 | 1024 | 1000 | agg | output_token_throughput | 90.89040550002517 | 995.5337333705638 | agg_patternA|tp=8|pp=1|dp=2|moe_tp=1|moe_ep=16|cp=1|bs=13|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | decode_smoke | 4096 | 1024 | 1000 | disagg | output_token_throughput | 184.03199999999998 | 938.959 | disagg_AA|p_tp=4|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=1|p_workers=3|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=33|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | decode_smoke | 4096 | 1024 | 2000 | agg | output_token_throughput | 204.46675246225885 | 1998.6991583970541 | agg_patternA|tp=4|pp=1|dp=4|moe_tp=1|moe_ep=16|cp=1|bs=182|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | decode_smoke | 4096 | 1024 | 2000 | disagg | output_token_throughput | 184.07520000000002 | 1877.68 | disagg_AA|p_tp=4|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=2|p_workers=3|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=33|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | decode_smoke | 4096 | 1024 | 5000 | agg | output_token_throughput | 209.09369382095153 | 2088.5551334987285 | agg_patternA|tp=4|pp=1|dp=4|moe_tp=1|moe_ep=16|cp=1|bs=250|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | decode_smoke | 4096 | 1024 | 5000 | disagg | output_token_throughput | 184.07520000000002 | 1877.68 | disagg_AA|p_tp=4|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=2|p_workers=3|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=33|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | primary | 16384 | 1 | 5000 | agg | prefill_input_throughput | 2463.76529592741 | 4987.488061590929 | agg_patternA|tp=8|pp=1|dp=4|moe_tp=1|moe_ep=32|cp=1|bs=6|ctx_tokens=16384 |
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | primary | 16384 | 1 | 5000 | disagg | prefill_input_throughput | 275.40540940331965 | 3718.155 | disagg_AA|p_tp=8|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=32|p_cp=1|p_bs=1|p_workers=1|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | primary | 4096 | 1 | 1000 | agg | prefill_input_throughput | 5318.5408478988975 | 962.6700530132562 | agg_patternA|tp=8|pp=1|dp=2|moe_tp=1|moe_ep=16|cp=1|bs=10|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | primary | 4096 | 1 | 1000 | disagg | prefill_input_throughput | 817.9270873382118 | 938.959 | disagg_AA|p_tp=4|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=1|p_workers=3|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | primary | 4096 | 1 | 2000 | agg | prefill_input_throughput | 145221.50251240918 | 1487.8237469106018 | agg_patternA|tp=8|pp=1|dp=2|moe_tp=1|moe_ep=16|cp=1|bs=422|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | primary | 4096 | 1 | 2000 | disagg | prefill_input_throughput | 818.0307613650888 | 1877.68 | disagg_AA|p_tp=4|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=2|p_workers=3|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | primary | 4096 | 1 | 5000 | agg | prefill_input_throughput | 230889.40823872894 | 3774.205177480415 | agg_patternA|tp=4|pp=1|dp=8|moe_tp=1|moe_ep=32|cp=1|bs=851|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h100_sxm | primary | 4096 | 1 | 5000 | disagg | prefill_input_throughput | 818.0307613650888 | 1877.68 | disagg_AA|p_tp=4|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=2|p_workers=3|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | decode_smoke | 4096 | 1024 | 1000 | agg | output_token_throughput | 1310.911657779627 | 720.2390901226229 | agg_patternA|tp=2|pp=1|dp=4|moe_tp=1|moe_ep=8|cp=1|bs=323|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | decode_smoke | 4096 | 1024 | 1000 | disagg | output_token_throughput | 1200.0096 | 288.001 | disagg_AA|p_tp=2|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=3|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=178|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | decode_smoke | 4096 | 1024 | 200 | agg | output_token_throughput | 161.38673657112642 | 196.63706804953333 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=4|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | decode_smoke | 4096 | 1024 | 200 | disagg | output_token_throughput | 976.6656 | 176.93 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=3|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=135|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | decode_smoke | 4096 | 1024 | 2000 | agg | output_token_throughput | 1310.911657779627 | 720.2390901226229 | agg_patternA|tp=2|pp=1|dp=4|moe_tp=1|moe_ep=8|cp=1|bs=323|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | decode_smoke | 4096 | 1024 | 2000 | disagg | output_token_throughput | 1200.0096 | 288.001 | disagg_AA|p_tp=2|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=3|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=178|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | decode_smoke | 4096 | 1024 | 500 | agg | output_token_throughput | 1091.1324851403099 | 499.962818936633 | agg_patternA|tp=4|pp=1|dp=2|moe_tp=1|moe_ep=8|cp=1|bs=549|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | decode_smoke | 4096 | 1024 | 500 | disagg | output_token_throughput | 1200.0096 | 288.001 | disagg_AA|p_tp=2|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=3|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=178|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | decode_smoke | 4096 | 1024 | 5000 | agg | output_token_throughput | 1310.911657779627 | 720.2390901226229 | agg_patternA|tp=2|pp=1|dp=4|moe_tp=1|moe_ep=8|cp=1|bs=323|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | decode_smoke | 4096 | 1024 | 5000 | disagg | output_token_throughput | 1200.0096 | 288.001 | disagg_AA|p_tp=2|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=3|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=178|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | primary | 16384 | 1 | 1000 | agg | prefill_input_throughput | 256791.23016936288 | 933.1159784622115 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=117|ctx_tokens=16384 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | primary | 16384 | 1 | 1000 | disagg | prefill_input_throughput | 3990.0071695441325 | 449.122 | disagg_AA|p_tp=8|p_pp=1|p_dp=1|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | primary | 16384 | 1 | 2000 | agg | prefill_input_throughput | 256791.23016936288 | 933.1159784622115 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=117|ctx_tokens=16384 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | primary | 16384 | 1 | 2000 | disagg | prefill_input_throughput | 3990.0071695441325 | 449.122 | disagg_AA|p_tp=8|p_pp=1|p_dp=1|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | primary | 16384 | 1 | 500 | agg | prefill_input_throughput | 4113.0963240496785 | 497.9217209247308 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=1|ctx_tokens=16384 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | primary | 16384 | 1 | 500 | disagg | prefill_input_throughput | 3990.0071695441325 | 449.122 | disagg_AA|p_tp=8|p_pp=1|p_dp=1|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | primary | 16384 | 1 | 5000 | agg | prefill_input_throughput | 256791.23016936288 | 933.1159784622115 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=117|ctx_tokens=16384 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | primary | 16384 | 1 | 5000 | disagg | prefill_input_throughput | 3990.0071695441325 | 449.122 | disagg_AA|p_tp=8|p_pp=1|p_dp=1|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | primary | 4096 | 1 | 1000 | agg | prefill_input_throughput | 1368381.9924646383 | 511.85707197042063 | agg_patternA|tp=4|pp=1|dp=2|moe_tp=1|moe_ep=8|cp=1|bs=684|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | primary | 4096 | 1 | 1000 | disagg | prefill_input_throughput | 6222.200617358969 | 288.001 | disagg_AA|p_tp=2|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | primary | 4096 | 1 | 200 | agg | prefill_input_throughput | 10415.139610309046 | 196.63682644953332 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=4|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | primary | 4096 | 1 | 200 | disagg | prefill_input_throughput | 5064.149663708811 | 176.93 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | primary | 4096 | 1 | 2000 | agg | prefill_input_throughput | 1368381.9924646383 | 511.85707197042063 | agg_patternA|tp=4|pp=1|dp=2|moe_tp=1|moe_ep=8|cp=1|bs=684|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | primary | 4096 | 1 | 2000 | disagg | prefill_input_throughput | 6222.200617358969 | 288.001 | disagg_AA|p_tp=2|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | primary | 4096 | 1 | 500 | agg | prefill_input_throughput | 1191767.725885426 | 358.2980061678996 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=834|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | primary | 4096 | 1 | 500 | disagg | prefill_input_throughput | 6222.200617358969 | 288.001 | disagg_AA|p_tp=2|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | primary | 4096 | 1 | 5000 | agg | prefill_input_throughput | 1368381.9924646383 | 511.85707197042063 | agg_patternA|tp=4|pp=1|dp=2|moe_tp=1|moe_ep=8|cp=1|bs=684|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h200_sxm | primary | 4096 | 1 | 5000 | disagg | prefill_input_throughput | 6222.200617358969 | 288.001 | disagg_AA|p_tp=2|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | decode_smoke | 4096 | 1024 | 1000 | agg | output_token_throughput | 90.89231278046095 | 995.4948873151652 | agg_patternA|tp=8|pp=1|dp=2|moe_tp=1|moe_ep=16|cp=1|bs=13|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | decode_smoke | 4096 | 1024 | 1000 | disagg | output_token_throughput | 184.03199999999998 | 938.912 | disagg_AA|p_tp=4|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=1|p_workers=3|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=33|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | decode_smoke | 4096 | 1024 | 2000 | agg | output_token_throughput | 204.47479049635606 | 1998.6047562815402 | agg_patternA|tp=4|pp=1|dp=4|moe_tp=1|moe_ep=16|cp=1|bs=182|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | decode_smoke | 4096 | 1024 | 2000 | disagg | output_token_throughput | 184.07520000000002 | 1877.585 | disagg_AA|p_tp=4|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=2|p_workers=3|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=33|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | decode_smoke | 4096 | 1024 | 5000 | agg | output_token_throughput | 209.10215789861894 | 2088.4570120611365 | agg_patternA|tp=4|pp=1|dp=4|moe_tp=1|moe_ep=16|cp=1|bs=250|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | decode_smoke | 4096 | 1024 | 5000 | disagg | output_token_throughput | 184.07520000000002 | 1877.585 | disagg_AA|p_tp=4|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=2|p_workers=3|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=33|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | primary | 16384 | 1 | 5000 | agg | prefill_input_throughput | 2463.8152397109134 | 4987.386960656103 | agg_patternA|tp=8|pp=1|dp=4|moe_tp=1|moe_ep=32|cp=1|bs=6|ctx_tokens=16384 |
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | primary | 16384 | 1 | 5000 | disagg | prefill_input_throughput | 275.411038872493 | 3718.079 | disagg_AA|p_tp=8|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=32|p_cp=1|p_bs=1|p_workers=1|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | primary | 4096 | 1 | 1000 | agg | prefill_input_throughput | 5318.748446563601 | 962.6324785688989 | agg_patternA|tp=8|pp=1|dp=2|moe_tp=1|moe_ep=16|cp=1|bs=10|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | primary | 4096 | 1 | 1000 | disagg | prefill_input_throughput | 817.9680310827852 | 938.912 | disagg_AA|p_tp=4|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=1|p_workers=3|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | primary | 4096 | 1 | 2000 | agg | prefill_input_throughput | 145226.96848683534 | 1487.767749001701 | agg_patternA|tp=8|pp=1|dp=2|moe_tp=1|moe_ep=16|cp=1|bs=422|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | primary | 4096 | 1 | 2000 | disagg | prefill_input_throughput | 818.0721511942202 | 1877.585 | disagg_AA|p_tp=4|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=2|p_workers=3|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=1|d_workers=1 |
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | primary | 4096 | 1 | 5000 | agg | prefill_input_throughput | 230896.1272202227 | 3774.0953496758243 | agg_patternA|tp=4|pp=1|dp=8|moe_tp=1|moe_ep=32|cp=1|bs=851|ctx_tokens=4096 |
| deepseek-ai/DeepSeek-V4-Pro | h800_sxm | primary | 4096 | 1 | 5000 | disagg | prefill_input_throughput | 818.0721511942202 | 1877.585 | disagg_AA|p_tp=4|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=16|p_cp=1|p_bs=2|p_workers=3|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | gb300 | decode_smoke | 4096 | 1024 | 1000 | agg | output_token_throughput | 6075.977171033757 | 388.43017820607145 | agg_patternA|tp=1|pp=1|dp=4|moe_tp=1|moe_ep=4|cp=1|bs=478|ctx_tokens=4096 |
| stepfun-ai/Step4 | gb300 | decode_smoke | 4096 | 1024 | 1000 | disagg | output_token_throughput | 5206.81728 | 96.679 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=1|p_workers=10|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=752|d_workers=3 |
| stepfun-ai/Step4 | gb300 | decode_smoke | 4096 | 1024 | 200 | agg | output_token_throughput | 532.2965004201199 | 199.75061847289408 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=57|ctx_tokens=4096 |
| stepfun-ai/Step4 | gb300 | decode_smoke | 4096 | 1024 | 200 | disagg | output_token_throughput | 5206.81728 | 96.679 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=1|p_workers=10|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=752|d_workers=3 |
| stepfun-ai/Step4 | gb300 | decode_smoke | 4096 | 1024 | 2000 | agg | output_token_throughput | 6075.977171033757 | 388.43017820607145 | agg_patternA|tp=1|pp=1|dp=4|moe_tp=1|moe_ep=4|cp=1|bs=478|ctx_tokens=4096 |
| stepfun-ai/Step4 | gb300 | decode_smoke | 4096 | 1024 | 2000 | disagg | output_token_throughput | 5206.81728 | 96.679 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=1|p_workers=10|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=752|d_workers=3 |
| stepfun-ai/Step4 | gb300 | decode_smoke | 4096 | 1024 | 500 | agg | output_token_throughput | 6075.977171033757 | 388.43017820607145 | agg_patternA|tp=1|pp=1|dp=4|moe_tp=1|moe_ep=4|cp=1|bs=478|ctx_tokens=4096 |
| stepfun-ai/Step4 | gb300 | decode_smoke | 4096 | 1024 | 500 | disagg | output_token_throughput | 5206.81728 | 96.679 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=1|p_workers=10|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=752|d_workers=3 |
| stepfun-ai/Step4 | gb300 | decode_smoke | 4096 | 1024 | 5000 | agg | output_token_throughput | 6075.977171033757 | 388.43017820607145 | agg_patternA|tp=1|pp=1|dp=4|moe_tp=1|moe_ep=4|cp=1|bs=478|ctx_tokens=4096 |
| stepfun-ai/Step4 | gb300 | decode_smoke | 4096 | 1024 | 5000 | disagg | output_token_throughput | 5206.81728 | 96.679 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=1|p_workers=10|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=752|d_workers=3 |
| stepfun-ai/Step4 | gb300 | primary | 16384 | 1 | 1000 | agg | prefill_input_throughput | 1351468.079138877 | 945.6028001891693 | agg_patternA|tp=2|pp=1|dp=2|moe_tp=1|moe_ep=4|cp=1|bs=156|ctx_tokens=16384 |
| stepfun-ai/Step4 | gb300 | primary | 16384 | 1 | 1000 | disagg | prefill_input_throughput | 22414.718998809218 | 685.264 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=1|p_workers=15|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | gb300 | primary | 16384 | 1 | 200 | disagg | prefill_input_throughput | 10624.862890650476 | 168.661 | disagg_AA|p_tp=8|p_pp=1|p_dp=1|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=2 |
| stepfun-ai/Step4 | gb300 | primary | 16384 | 1 | 2000 | agg | prefill_input_throughput | 2115762.6270837123 | 1773.3255857588938 | agg_patternA|tp=1|pp=1|dp=8|moe_tp=1|moe_ep=8|cp=1|bs=229|ctx_tokens=16384 |
| stepfun-ai/Step4 | gb300 | primary | 16384 | 1 | 2000 | disagg | prefill_input_throughput | 22416.894399060126 | 1370.395 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=2|p_workers=15|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | gb300 | primary | 16384 | 1 | 500 | agg | prefill_input_throughput | 655919.9519112392 | 499.5731552992041 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=160|ctx_tokens=16384 |
| stepfun-ai/Step4 | gb300 | primary | 16384 | 1 | 500 | disagg | prefill_input_throughput | 19116.69898417645 | 401.743 | disagg_AA|p_tp=2|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=1|p_workers=15|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | gb300 | primary | 16384 | 1 | 5000 | agg | prefill_input_throughput | 2115762.6270837123 | 1773.3255857588938 | agg_patternA|tp=1|pp=1|dp=8|moe_tp=1|moe_ep=8|cp=1|bs=229|ctx_tokens=16384 |
| stepfun-ai/Step4 | gb300 | primary | 16384 | 1 | 5000 | disagg | prefill_input_throughput | 22418.434491862372 | 4796.053 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=7|p_workers=15|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | gb300 | primary | 4096 | 1 | 1000 | agg | prefill_input_throughput | 7743038.773604454 | 494.0778564923961 | agg_patternA|tp=1|pp=1|dp=8|moe_tp=1|moe_ep=8|cp=1|bs=934|ctx_tokens=4096 |
| stepfun-ai/Step4 | gb300 | primary | 4096 | 1 | 1000 | disagg | prefill_input_throughput | 40239.00291104037 | 954.298 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=10|p_workers=15|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | gb300 | primary | 4096 | 1 | 200 | agg | prefill_input_throughput | 72247.14522329136 | 198.4299857896433 | agg_patternA|tp=2|pp=1|dp=2|moe_tp=1|moe_ep=4|cp=1|bs=7|ctx_tokens=4096 |
| stepfun-ai/Step4 | gb300 | primary | 4096 | 1 | 200 | disagg | prefill_input_throughput | 40216.372462257874 | 190.967 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=2|p_workers=15|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | gb300 | primary | 4096 | 1 | 2000 | agg | prefill_input_throughput | 7743038.773604454 | 494.0778564923961 | agg_patternA|tp=1|pp=1|dp=8|moe_tp=1|moe_ep=8|cp=1|bs=934|ctx_tokens=4096 |
| stepfun-ai/Step4 | gb300 | primary | 4096 | 1 | 2000 | disagg | prefill_input_throughput | 40241.786064604836 | 1908.464 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=20|p_workers=15|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | gb300 | primary | 4096 | 1 | 500 | agg | prefill_input_throughput | 7743038.773604454 | 494.0778564923961 | agg_patternA|tp=1|pp=1|dp=8|moe_tp=1|moe_ep=8|cp=1|bs=934|ctx_tokens=4096 |
| stepfun-ai/Step4 | gb300 | primary | 4096 | 1 | 500 | disagg | prefill_input_throughput | 40233.437758662236 | 477.215 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=5|p_workers=15|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | gb300 | primary | 4096 | 1 | 5000 | agg | prefill_input_throughput | 7743038.773604454 | 494.0778564923961 | agg_patternA|tp=1|pp=1|dp=8|moe_tp=1|moe_ep=8|cp=1|bs=934|ctx_tokens=4096 |
| stepfun-ai/Step4 | gb300 | primary | 4096 | 1 | 5000 | disagg | prefill_input_throughput | 40243.49247732071 | 4961.796 | disagg_AA|p_tp=1|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=52|p_workers=15|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | gb300 | primary | 65536 | 1 | 2000 | agg | prefill_input_throughput | 42652.277205623905 | 1920.6477442005948 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=10|ctx_tokens=65536 |
| stepfun-ai/Step4 | gb300 | primary | 65536 | 1 | 2000 | disagg | prefill_input_throughput | 5603.186812499609 | 1279.272 | disagg_AA|p_tp=8|p_pp=1|p_dp=1|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=2 |
| stepfun-ai/Step4 | gb300 | primary | 65536 | 1 | 5000 | agg | prefill_input_throughput | 241577.64097795196 | 4679.638378053277 | agg_patternA|tp=4|pp=1|dp=4|moe_tp=1|moe_ep=16|cp=1|bs=69|ctx_tokens=65536 |
| stepfun-ai/Step4 | gb300 | primary | 65536 | 1 | 5000 | disagg | prefill_input_throughput | 7614.337758340946 | 4034.494 | disagg_AA|p_tp=2|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=4|p_cp=1|p_bs=1|p_workers=15|d_tp=1|d_pp=1|d_dp=4|d_moe_tp=1|d_moe_ep=4|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h100_sxm | decode_smoke | 4096 | 1024 | 1000 | agg | output_token_throughput | 200.38637645713789 | 948.4342626723804 | agg_patternA|tp=8|pp=1|dp=2|moe_tp=1|moe_ep=16|cp=1|bs=152|ctx_tokens=4096 |
| stepfun-ai/Step4 | h100_sxm | decode_smoke | 4096 | 1024 | 1000 | disagg | output_token_throughput | 690.72128 | 124.503 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=2|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=106|d_workers=1 |
| stepfun-ai/Step4 | h100_sxm | decode_smoke | 4096 | 1024 | 200 | agg | output_token_throughput | 125.83843317410329 | 195.32195696234695 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=3|ctx_tokens=4096 |
| stepfun-ai/Step4 | h100_sxm | decode_smoke | 4096 | 1024 | 200 | disagg | output_token_throughput | 690.72128 | 124.503 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=2|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=106|d_workers=1 |
| stepfun-ai/Step4 | h100_sxm | decode_smoke | 4096 | 1024 | 2000 | agg | output_token_throughput | 278.97553734965277 | 1995.1576831814016 | agg_patternA|tp=2|pp=1|dp=8|moe_tp=1|moe_ep=16|cp=1|bs=39|ctx_tokens=4096 |
| stepfun-ai/Step4 | h100_sxm | decode_smoke | 4096 | 1024 | 2000 | disagg | output_token_throughput | 690.72128 | 124.503 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=2|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=106|d_workers=1 |
| stepfun-ai/Step4 | h100_sxm | decode_smoke | 4096 | 1024 | 500 | agg | output_token_throughput | 187.2731662028338 | 286.652797964252 | agg_patternA|tp=4|pp=1|dp=2|moe_tp=1|moe_ep=8|cp=1|bs=11|ctx_tokens=4096 |
| stepfun-ai/Step4 | h100_sxm | decode_smoke | 4096 | 1024 | 500 | disagg | output_token_throughput | 690.72128 | 124.503 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=2|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=106|d_workers=1 |
| stepfun-ai/Step4 | h100_sxm | decode_smoke | 4096 | 1024 | 5000 | agg | output_token_throughput | 382.0630632084591 | 3836.2744820888493 | agg_patternA|tp=1|pp=1|dp=16|moe_tp=1|moe_ep=16|cp=1|bs=101|ctx_tokens=4096 |
| stepfun-ai/Step4 | h100_sxm | decode_smoke | 4096 | 1024 | 5000 | disagg | output_token_throughput | 690.72128 | 124.503 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=2|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=106|d_workers=1 |
| stepfun-ai/Step4 | h100_sxm | primary | 16384 | 1 | 1000 | agg | prefill_input_throughput | 11156.74819704064 | 550.6980969266398 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=3|ctx_tokens=16384 |
| stepfun-ai/Step4 | h100_sxm | primary | 16384 | 1 | 1000 | disagg | prefill_input_throughput | 5545.37802643957 | 646.304 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=4|d_pp=1|d_dp=2|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h100_sxm | primary | 16384 | 1 | 2000 | agg | prefill_input_throughput | 11156.74819704064 | 550.6980969266398 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=3|ctx_tokens=16384 |
| stepfun-ai/Step4 | h100_sxm | primary | 16384 | 1 | 2000 | disagg | prefill_input_throughput | 5545.37802643957 | 646.304 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=4|d_pp=1|d_dp=2|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h100_sxm | primary | 16384 | 1 | 500 | agg | prefill_input_throughput | 4467.414624497842 | 458.4306969783905 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=1|ctx_tokens=16384 |
| stepfun-ai/Step4 | h100_sxm | primary | 16384 | 1 | 500 | disagg | prefill_input_throughput | 4656.589515917376 | 384.831 | disagg_AA|p_tp=8|p_pp=1|p_dp=1|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=4|d_pp=1|d_dp=2|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h100_sxm | primary | 16384 | 1 | 5000 | agg | prefill_input_throughput | 28220.633705692715 | 4789.686915242221 | agg_patternA|tp=8|pp=1|dp=4|moe_tp=1|moe_ep=32|cp=1|bs=66|ctx_tokens=16384 |
| stepfun-ai/Step4 | h100_sxm | primary | 16384 | 1 | 5000 | disagg | prefill_input_throughput | 5545.37802643957 | 646.304 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=4|d_pp=1|d_dp=2|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h100_sxm | primary | 4096 | 1 | 1000 | agg | prefill_input_throughput | 99942.2115263856 | 978.4854518071384 | agg_patternA|tp=8|pp=1|dp=2|moe_tp=1|moe_ep=16|cp=1|bs=191|ctx_tokens=4096 |
| stepfun-ai/Step4 | h100_sxm | primary | 4096 | 1 | 1000 | disagg | prefill_input_throughput | 7885.085440092931 | 454.529 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=4|p_workers=7|d_tp=4|d_pp=1|d_dp=2|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h100_sxm | primary | 4096 | 1 | 200 | agg | prefill_input_throughput | 7864.720675316047 | 195.30254962783852 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=3|ctx_tokens=4096 |
| stepfun-ai/Step4 | h100_sxm | primary | 4096 | 1 | 200 | disagg | prefill_input_throughput | 7196.613736215191 | 124.503 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=4|d_pp=1|d_dp=2|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h100_sxm | primary | 4096 | 1 | 2000 | agg | prefill_input_throughput | 128781.24610372537 | 1447.1672362130435 | agg_patternA|tp=4|pp=1|dp=4|moe_tp=1|moe_ep=16|cp=1|bs=182|ctx_tokens=4096 |
| stepfun-ai/Step4 | h100_sxm | primary | 4096 | 1 | 2000 | disagg | prefill_input_throughput | 7885.085440092931 | 454.529 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=4|p_workers=7|d_tp=4|d_pp=1|d_dp=2|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h100_sxm | primary | 4096 | 1 | 500 | agg | prefill_input_throughput | 33025.41492283151 | 279.05781112923097 | agg_patternA|tp=4|pp=1|dp=2|moe_tp=1|moe_ep=8|cp=1|bs=9|ctx_tokens=4096 |
| stepfun-ai/Step4 | h100_sxm | primary | 4096 | 1 | 500 | disagg | prefill_input_throughput | 7885.085440092931 | 454.529 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=4|p_workers=7|d_tp=4|d_pp=1|d_dp=2|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h100_sxm | primary | 4096 | 1 | 5000 | agg | prefill_input_throughput | 140100.35009451074 | 2382.74922064652 | agg_patternA|tp=2|pp=1|dp=8|moe_tp=1|moe_ep=16|cp=1|bs=163|ctx_tokens=4096 |
| stepfun-ai/Step4 | h100_sxm | primary | 4096 | 1 | 5000 | disagg | prefill_input_throughput | 7885.085440092931 | 454.529 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=4|p_workers=7|d_tp=4|d_pp=1|d_dp=2|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h200_sxm | decode_smoke | 4096 | 1024 | 1000 | agg | output_token_throughput | 2266.4643724199677 | 740.0504923149001 | agg_patternA|tp=1|pp=1|dp=8|moe_tp=1|moe_ep=8|cp=1|bs=208|ctx_tokens=4096 |
| stepfun-ai/Step4 | h200_sxm | decode_smoke | 4096 | 1024 | 1000 | disagg | output_token_throughput | 1971.78816 | 279.399 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=5|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=213|d_workers=3 |
| stepfun-ai/Step4 | h200_sxm | decode_smoke | 4096 | 1024 | 200 | agg | output_token_throughput | 198.2361728390824 | 196.9689232997745 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=5|ctx_tokens=4096 |
| stepfun-ai/Step4 | h200_sxm | decode_smoke | 4096 | 1024 | 200 | disagg | output_token_throughput | 1687.68 | 170.65 | disagg_AA|p_tp=2|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=5|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=165|d_workers=3 |
| stepfun-ai/Step4 | h200_sxm | decode_smoke | 4096 | 1024 | 2000 | agg | output_token_throughput | 2266.4643724199677 | 740.0504923149001 | agg_patternA|tp=1|pp=1|dp=8|moe_tp=1|moe_ep=8|cp=1|bs=208|ctx_tokens=4096 |
| stepfun-ai/Step4 | h200_sxm | decode_smoke | 4096 | 1024 | 2000 | disagg | output_token_throughput | 1971.78816 | 279.399 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=5|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=213|d_workers=3 |
| stepfun-ai/Step4 | h200_sxm | decode_smoke | 4096 | 1024 | 500 | agg | output_token_throughput | 1517.175389970408 | 499.64708862707 | agg_patternA|tp=2|pp=1|dp=4|moe_tp=1|moe_ep=8|cp=1|bs=169|ctx_tokens=4096 |
| stepfun-ai/Step4 | h200_sxm | decode_smoke | 4096 | 1024 | 500 | disagg | output_token_throughput | 1971.78816 | 279.399 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=5|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=213|d_workers=3 |
| stepfun-ai/Step4 | h200_sxm | decode_smoke | 4096 | 1024 | 5000 | agg | output_token_throughput | 2266.4643724199677 | 740.0504923149001 | agg_patternA|tp=1|pp=1|dp=8|moe_tp=1|moe_ep=8|cp=1|bs=208|ctx_tokens=4096 |
| stepfun-ai/Step4 | h200_sxm | decode_smoke | 4096 | 1024 | 5000 | disagg | output_token_throughput | 1971.78816 | 279.399 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=5|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=213|d_workers=3 |
| stepfun-ai/Step4 | h200_sxm | primary | 16384 | 1 | 1000 | agg | prefill_input_throughput | 194054.80342102124 | 823.1901358989784 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=78|ctx_tokens=16384 |
| stepfun-ai/Step4 | h200_sxm | primary | 16384 | 1 | 1000 | disagg | prefill_input_throughput | 5655.28928919136 | 633.743 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h200_sxm | primary | 16384 | 1 | 2000 | agg | prefill_input_throughput | 241651.5541363179 | 1288.2019365139072 | agg_patternA|tp=4|pp=1|dp=2|moe_tp=1|moe_ep=8|cp=1|bs=76|ctx_tokens=16384 |
| stepfun-ai/Step4 | h200_sxm | primary | 16384 | 1 | 2000 | disagg | prefill_input_throughput | 7609.188349704199 | 1884.038 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h200_sxm | primary | 16384 | 1 | 500 | agg | prefill_input_throughput | 4596.783293444625 | 445.52894258047996 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=1|ctx_tokens=16384 |
| stepfun-ai/Step4 | h200_sxm | primary | 16384 | 1 | 500 | disagg | prefill_input_throughput | 4818.123889236925 | 371.929 | disagg_AA|p_tp=8|p_pp=1|p_dp=1|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h200_sxm | primary | 16384 | 1 | 5000 | agg | prefill_input_throughput | 255979.90121312992 | 2016.1582903741194 | agg_patternA|tp=2|pp=1|dp=4|moe_tp=1|moe_ep=8|cp=1|bs=63|ctx_tokens=16384 |
| stepfun-ai/Step4 | h200_sxm | primary | 16384 | 1 | 5000 | disagg | prefill_input_throughput | 7609.630620270355 | 3767.857 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=2|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h200_sxm | primary | 4096 | 1 | 1000 | agg | prefill_input_throughput | 1392281.3918962714 | 761.9609126249366 | agg_patternA|tp=1|pp=1|dp=8|moe_tp=1|moe_ep=8|cp=1|bs=259|ctx_tokens=4096 |
| stepfun-ai/Step4 | h200_sxm | primary | 4096 | 1 | 1000 | disagg | prefill_input_throughput | 12834.255238386264 | 837.758 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=3|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h200_sxm | primary | 4096 | 1 | 200 | agg | prefill_input_throughput | 12998.899340041085 | 196.93975105371564 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=5|ctx_tokens=4096 |
| stepfun-ai/Step4 | h200_sxm | primary | 4096 | 1 | 200 | disagg | prefill_input_throughput | 10501.025490770584 | 170.65 | disagg_AA|p_tp=2|p_pp=1|p_dp=4|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h200_sxm | primary | 4096 | 1 | 2000 | agg | prefill_input_throughput | 1392281.3918962714 | 761.9609126249366 | agg_patternA|tp=1|pp=1|dp=8|moe_tp=1|moe_ep=8|cp=1|bs=259|ctx_tokens=4096 |
| stepfun-ai/Step4 | h200_sxm | primary | 4096 | 1 | 2000 | disagg | prefill_input_throughput | 12836.183629875031 | 1954.475 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=7|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h200_sxm | primary | 4096 | 1 | 500 | agg | prefill_input_throughput | 772848.740472446 | 417.3649811511857 | agg_patternA|tp=4|pp=1|dp=2|moe_tp=1|moe_ep=8|cp=1|bs=315|ctx_tokens=4096 |
| stepfun-ai/Step4 | h200_sxm | primary | 4096 | 1 | 500 | disagg | prefill_input_throughput | 12827.533384156708 | 279.399 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h200_sxm | primary | 4096 | 1 | 5000 | agg | prefill_input_throughput | 1392281.3918962714 | 761.9609126249366 | agg_patternA|tp=1|pp=1|dp=8|moe_tp=1|moe_ep=8|cp=1|bs=259|ctx_tokens=4096 |
| stepfun-ai/Step4 | h200_sxm | primary | 4096 | 1 | 5000 | disagg | prefill_input_throughput | 12837.03051807641 | 4746.269 | disagg_AA|p_tp=1|p_pp=1|p_dp=8|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=17|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h200_sxm | primary | 65536 | 1 | 5000 | agg | prefill_input_throughput | 18657.329030752135 | 4390.767824535576 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=10|ctx_tokens=65536 |
| stepfun-ai/Step4 | h200_sxm | primary | 65536 | 1 | 5000 | disagg | prefill_input_throughput | 2372.10238163315 | 3021.792 | disagg_AA|p_tp=8|p_pp=1|p_dp=1|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=1|d_pp=1|d_dp=8|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h800_sxm | decode_smoke | 4096 | 1024 | 1000 | agg | output_token_throughput | 188.4947719565686 | 999.284585151032 | agg_patternA|tp=8|pp=1|dp=2|moe_tp=1|moe_ep=16|cp=1|bs=143|ctx_tokens=4096 |
| stepfun-ai/Step4 | h800_sxm | decode_smoke | 4096 | 1024 | 1000 | disagg | output_token_throughput | 631.1232000000001 | 730.137 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=4|p_workers=2|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=80|d_workers=1 |
| stepfun-ai/Step4 | h800_sxm | decode_smoke | 4096 | 1024 | 200 | disagg | output_token_throughput | 595.6415999999999 | 193.412 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=2|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=66|d_workers=1 |
| stepfun-ai/Step4 | h800_sxm | decode_smoke | 4096 | 1024 | 2000 | agg | output_token_throughput | 269.44971807606487 | 1997.9554134072716 | agg_patternA|tp=2|pp=1|dp=8|moe_tp=1|moe_ep=16|cp=1|bs=35|ctx_tokens=4096 |
| stepfun-ai/Step4 | h800_sxm | decode_smoke | 4096 | 1024 | 2000 | disagg | output_token_throughput | 631.1232000000001 | 730.137 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=4|p_workers=2|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=80|d_workers=1 |
| stepfun-ai/Step4 | h800_sxm | decode_smoke | 4096 | 1024 | 500 | agg | output_token_throughput | 175.45396140124043 | 385.6011476162174 | agg_patternA|tp=4|pp=1|dp=2|moe_tp=1|moe_ep=8|cp=1|bs=11|ctx_tokens=4096 |
| stepfun-ai/Step4 | h800_sxm | decode_smoke | 4096 | 1024 | 500 | disagg | output_token_throughput | 631.0656 | 365.108 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=2|p_workers=2|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=80|d_workers=1 |
| stepfun-ai/Step4 | h800_sxm | decode_smoke | 4096 | 1024 | 5000 | agg | output_token_throughput | 382.07633435799437 | 3836.106949745501 | agg_patternA|tp=1|pp=1|dp=16|moe_tp=1|moe_ep=16|cp=1|bs=101|ctx_tokens=4096 |
| stepfun-ai/Step4 | h800_sxm | decode_smoke | 4096 | 1024 | 5000 | disagg | output_token_throughput | 631.1232000000001 | 730.137 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=4|p_workers=2|d_tp=1|d_pp=1|d_dp=16|d_moe_tp=1|d_moe_ep=16|d_cp=1|d_bs=80|d_workers=1 |
| stepfun-ai/Step4 | h800_sxm | primary | 16384 | 1 | 1000 | agg | prefill_input_throughput | 8041.777317217665 | 764.0102128724117 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=3|ctx_tokens=16384 |
| stepfun-ai/Step4 | h800_sxm | primary | 16384 | 1 | 1000 | disagg | prefill_input_throughput | 3887.9776441285458 | 921.816 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=4|d_pp=1|d_dp=2|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h800_sxm | primary | 16384 | 1 | 2000 | agg | prefill_input_throughput | 8041.777317217665 | 764.0102128724117 | agg_patternA|tp=8|pp=1|dp=1|moe_tp=1|moe_ep=8|cp=1|bs=3|ctx_tokens=16384 |
| stepfun-ai/Step4 | h800_sxm | primary | 16384 | 1 | 2000 | disagg | prefill_input_throughput | 3887.9776441285458 | 921.816 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=4|d_pp=1|d_dp=2|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h800_sxm | primary | 16384 | 1 | 5000 | agg | prefill_input_throughput | 26657.745364540093 | 4993.6706266642905 | agg_patternA|tp=8|pp=1|dp=4|moe_tp=1|moe_ep=32|cp=1|bs=65|ctx_tokens=16384 |
| stepfun-ai/Step4 | h800_sxm | primary | 16384 | 1 | 5000 | disagg | prefill_input_throughput | 3887.9776441285458 | 921.816 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=4|d_pp=1|d_dp=2|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h800_sxm | primary | 4096 | 1 | 1000 | agg | prefill_input_throughput | 74310.76900320637 | 999.0476615414474 | agg_patternA|tp=8|pp=1|dp=2|moe_tp=1|moe_ep=16|cp=1|bs=145|ctx_tokens=4096 |
| stepfun-ai/Step4 | h800_sxm | primary | 4096 | 1 | 1000 | disagg | prefill_input_throughput | 4908.667825353325 | 730.137 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=4|p_workers=7|d_tp=4|d_pp=1|d_dp=2|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h800_sxm | primary | 4096 | 1 | 200 | disagg | prefill_input_throughput | 4632.597770562323 | 193.412 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=1|p_workers=7|d_tp=4|d_pp=1|d_dp=2|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h800_sxm | primary | 4096 | 1 | 2000 | agg | prefill_input_throughput | 124288.97008960613 | 1499.4733632890996 | agg_patternA|tp=4|pp=1|dp=4|moe_tp=1|moe_ep=16|cp=1|bs=182|ctx_tokens=4096 |
| stepfun-ai/Step4 | h800_sxm | primary | 4096 | 1 | 2000 | disagg | prefill_input_throughput | 4908.667825353325 | 730.137 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=4|p_workers=7|d_tp=4|d_pp=1|d_dp=2|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h800_sxm | primary | 4096 | 1 | 500 | agg | prefill_input_throughput | 26896.81799696142 | 380.7141796905801 | agg_patternA|tp=4|pp=1|dp=2|moe_tp=1|moe_ep=8|cp=1|bs=10|ctx_tokens=4096 |
| stepfun-ai/Step4 | h800_sxm | primary | 4096 | 1 | 500 | disagg | prefill_input_throughput | 4908.1367704898275 | 365.108 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=2|p_workers=7|d_tp=4|d_pp=1|d_dp=2|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |
| stepfun-ai/Step4 | h800_sxm | primary | 4096 | 1 | 5000 | agg | prefill_input_throughput | 138112.15631820148 | 2417.050091020888 | agg_patternA|tp=2|pp=1|dp=8|moe_tp=1|moe_ep=16|cp=1|bs=163|ctx_tokens=4096 |
| stepfun-ai/Step4 | h800_sxm | primary | 4096 | 1 | 5000 | disagg | prefill_input_throughput | 4908.667825353325 | 730.137 | disagg_AA|p_tp=4|p_pp=1|p_dp=2|p_moe_tp=1|p_moe_ep=8|p_cp=1|p_bs=4|p_workers=7|d_tp=4|d_pp=1|d_dp=2|d_moe_tp=1|d_moe_ep=8|d_cp=1|d_bs=1|d_workers=1 |

## Paired Model Deltas

Absolute delta is Step4 minus DeepSeek-V4-Pro. TPOT is observational and does not affect eligibility.

| Metric | Step4 | DeepSeek-V4-Pro | Absolute delta | Relative delta | Polarity | Status |
|---|---:|---:|---:|---:|---|---|
| ranking_metric_value | 6075.977171033757 | 5686.8983915650015 | 389.0787794687558 | 0.0684166926642914 | higher_is_better | computed |
| ttft | 388.43017820607145 | 421.9725282053259 | -33.54234999925444 | -0.07948941638905271 | lower_is_better | computed |
| tpot | 78.29077618816083 | 127.42521278683706 | -49.13443659867623 | -0.38559430684154045 | lower_is_better | computed |
| request_latency | 80479.8942186946 | 130777.96520913964 | -50298.07099044505 | -0.3846066186303523 | lower_is_better | computed |
| ranking_metric_value | 5206.81728 | 5059.2384 | 147.57888000000003 | 0.02917017707645483 | higher_is_better | computed |
| ttft | 96.679 | 273.243 | -176.564 | -0.6461794080726679 | lower_is_better | computed |
| tpot | 49.876 | 54.964 | -5.088000000000001 | -0.0925696819736555 | lower_is_better | computed |
| request_latency | 51119.827 | 56501.415 | -5381.588000000003 | -0.09524695974428257 | lower_is_better | computed |
| ranking_metric_value | 532.2965004201199 | 940.3004636798224 | -408.0039632597026 | -0.43390807408835885 | higher_is_better | computed |
| ttft | 199.75061847289408 | 199.89558104822441 | -0.1449625753303394 | -0.000725191495330592 | lower_is_better | computed |
| tpot | 13.190137391283885 | 20.80852320695589 | -7.618385815672005 | -0.36611852460175187 | lower_is_better | computed |
| request_latency | 13693.261169756308 | 21487.0148217641 | -7793.753652007792 | -0.3627192384171269 | lower_is_better | computed |
| ranking_metric_value | 5206.81728 | 5054.2272 | 152.59007999999994 | 0.030190585813000242 | higher_is_better | computed |
| ttft | 96.679 | 136.758 | -40.07900000000001 | -0.29306512233288 | lower_is_better | computed |
| tpot | 49.876 | 45.492 | 4.384 | 0.09636859227996132 | lower_is_better | computed |
| request_latency | 51119.827 | 46675.074 | 4444.752999999997 | 0.09522755121930812 | lower_is_better | computed |
| ranking_metric_value | 6075.977171033757 | 5686.8983915650015 | 389.0787794687558 | 0.0684166926642914 | higher_is_better | computed |
| ttft | 388.43017820607145 | 421.9725282053259 | -33.54234999925444 | -0.07948941638905271 | lower_is_better | computed |
| tpot | 78.29077618816083 | 127.42521278683706 | -49.13443659867623 | -0.38559430684154045 | lower_is_better | computed |
| request_latency | 80479.8942186946 | 130777.96520913964 | -50298.07099044505 | -0.3846066186303523 | lower_is_better | computed |
| ranking_metric_value | 5206.81728 | 5059.2384 | 147.57888000000003 | 0.02917017707645483 | higher_is_better | computed |
| ttft | 96.679 | 273.243 | -176.564 | -0.6461794080726679 | lower_is_better | computed |
| tpot | 49.876 | 54.964 | -5.088000000000001 | -0.0925696819736555 | lower_is_better | computed |
| request_latency | 51119.827 | 56501.415 | -5381.588000000003 | -0.09524695974428257 | lower_is_better | computed |
| ranking_metric_value | 6075.977171033757 | 5686.8983915650015 | 389.0787794687558 | 0.0684166926642914 | higher_is_better | computed |
| ttft | 388.43017820607145 | 421.9725282053259 | -33.54234999925444 | -0.07948941638905271 | lower_is_better | computed |
| tpot | 78.29077618816083 | 127.42521278683706 | -49.13443659867623 | -0.38559430684154045 | lower_is_better | computed |
| request_latency | 80479.8942186946 | 130777.96520913964 | -50298.07099044505 | -0.3846066186303523 | lower_is_better | computed |
| ranking_metric_value | 5206.81728 | 5059.2384 | 147.57888000000003 | 0.02917017707645483 | higher_is_better | computed |
| ttft | 96.679 | 273.243 | -176.564 | -0.6461794080726679 | lower_is_better | computed |
| tpot | 49.876 | 54.964 | -5.088000000000001 | -0.0925696819736555 | lower_is_better | computed |
| request_latency | 51119.827 | 56501.415 | -5381.588000000003 | -0.09524695974428257 | lower_is_better | computed |
| ranking_metric_value | 6075.977171033757 | 5686.8983915650015 | 389.0787794687558 | 0.0684166926642914 | higher_is_better | computed |
| ttft | 388.43017820607145 | 421.9725282053259 | -33.54234999925444 | -0.07948941638905271 | lower_is_better | computed |
| tpot | 78.29077618816083 | 127.42521278683706 | -49.13443659867623 | -0.38559430684154045 | lower_is_better | computed |
| request_latency | 80479.8942186946 | 130777.96520913964 | -50298.07099044505 | -0.3846066186303523 | lower_is_better | computed |
| ranking_metric_value | 5206.81728 | 5059.2384 | 147.57888000000003 | 0.02917017707645483 | higher_is_better | computed |
| ttft | 96.679 | 273.243 | -176.564 | -0.6461794080726679 | lower_is_better | computed |
| tpot | 49.876 | 54.964 | -5.088000000000001 | -0.0925696819736555 | lower_is_better | computed |
| request_latency | 51119.827 | 56501.415 | -5381.588000000003 | -0.09524695974428257 | lower_is_better | computed |
| ranking_metric_value | 1351468.079138877 | 8914346.540325223 | -7562878.461186346 | -0.8483940384160148 | higher_is_better | computed |
| ttft | 945.6028001891693 | 917.1301522794208 | 28.472647909748503 | 0.03104537326461575 | lower_is_better | computed |
| tpot | 421.305027240849 | 409.7650761397104 | 11.53995110113857 | 0.0281623588077672 | lower_is_better | computed |
| request_latency | 945.6028001891693 | 917.1301522794208 | 28.472647909748503 | 0.03104537326461575 | lower_is_better | computed |
| ranking_metric_value | 22414.718998809218 | 20449.880247806446 | 1964.8387510027715 | 0.09608069715780021 | higher_is_better | computed |
| ttft | 685.264 | 701.031 | -15.766999999999939 | -0.02249115944943938 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 685.264 | 701.031 | -15.766999999999939 | -0.02249115944943938 | lower_is_better | computed |
| ranking_metric_value | 10624.862890650476 | 9215.736693237337 | 1409.1261974131394 | 0.15290434658871926 | higher_is_better | computed |
| ttft | 168.661 | 194.45 | -25.788999999999987 | -0.13262535356132676 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 168.661 | 194.45 | -25.788999999999987 | -0.13262535356132676 | lower_is_better | computed |
| ranking_metric_value | 2115762.6270837123 | 9718746.339774225 | -7602983.712690513 | -0.7823008695653576 | higher_is_better | computed |
| ttft | 1773.3255857588938 | 1376.4672450861365 | 396.8583406727573 | 0.2883165887815389 | lower_is_better | computed |
| tpot | 822.0640040664348 | 639.4336225430683 | 182.6303815233665 | 0.2856127283345437 | lower_is_better | computed |
| request_latency | 1773.3255857588938 | 1376.4672450861365 | 396.8583406727573 | 0.2883165887815389 | lower_is_better | computed |
| ranking_metric_value | 22416.894399060126 | 20449.880247806446 | 1967.0141512536793 | 0.09618707432111594 | higher_is_better | computed |
| ttft | 1370.395 | 701.031 | 669.364 | 0.9548279605324159 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 1370.395 | 701.031 | 669.364 | 0.9548279605324159 | lower_is_better | computed |
| ranking_metric_value | 655919.9519112392 | 1982473.3251880198 | -1326553.3732767806 | -0.6691405914129861 | higher_is_better | computed |
| ttft | 499.5731552992041 | 499.9976480924355 | -0.42449279323142264 | -0.0008489895799528758 | lower_is_better | computed |
| tpot | 187.24088227298347 | 201.19882404621774 | -13.957941773234268 | -0.06937387352735205 | lower_is_better | computed |
| request_latency | 499.5731552992041 | 499.9976480924355 | -0.42449279323142264 | -0.0008489895799528758 | lower_is_better | computed |
| ranking_metric_value | 19116.69898417645 | 20312.57025425096 | -1195.871270074509 | -0.05887345890283089 | higher_is_better | computed |
| ttft | 401.743 | 378.091 | 23.651999999999987 | 0.06255636870488847 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 401.743 | 378.091 | 23.651999999999987 | 0.06255636870488847 | lower_is_better | computed |
| ranking_metric_value | 2115762.6270837123 | 9718746.339774225 | -7602983.712690513 | -0.7823008695653576 | higher_is_better | computed |
| ttft | 1773.3255857588938 | 1376.4672450861365 | 396.8583406727573 | 0.2883165887815389 | lower_is_better | computed |
| tpot | 822.0640040664348 | 639.4336225430683 | 182.6303815233665 | 0.2856127283345437 | lower_is_better | computed |
| request_latency | 1773.3255857588938 | 1376.4672450861365 | 396.8583406727573 | 0.2883165887815389 | lower_is_better | computed |
| ranking_metric_value | 22418.434491862372 | 20449.880247806446 | 1968.5542440559257 | 0.09626238492360278 | higher_is_better | computed |
| ttft | 4796.053 | 701.031 | 4095.022 | 5.841427839853017 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 4796.053 | 701.031 | 4095.022 | 5.841427839853017 | lower_is_better | computed |
| ranking_metric_value | 7743038.773604454 | 17173713.065674055 | -9430674.292069603 | -0.5491342644427404 | higher_is_better | computed |
| ttft | 494.0778564923961 | 677.828490291194 | -183.75063379879794 | -0.27108720927304036 | lower_is_better | computed |
| tpot | 173.43892824619803 | 290.114245145597 | -116.67531689939898 | -0.40217024448711297 | lower_is_better | computed |
| request_latency | 494.0778564923961 | 677.828490291194 | -183.75063379879794 | -0.27108720927304036 | lower_is_better | computed |
| ranking_metric_value | 40239.00291104037 | 28106.84994675069 | 12132.152964289678 | 0.43164399380487045 | higher_is_better | computed |
| ttft | 954.298 | 273.243 | 681.0550000000001 | 2.4924883711568095 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 954.298 | 273.243 | 681.0550000000001 | 2.4924883711568095 | lower_is_better | computed |
| ranking_metric_value | 72247.14522329136 | 404714.7927922588 | -332467.6475689674 | -0.8214862750016256 | higher_is_better | computed |
| ttft | 198.4299857896433 | 199.88397123285813 | -1.4539854432148331 | -0.007274147267771605 | lower_is_better | computed |
| tpot | 73.28514421574255 | 55.6886748036649 | 17.596469412077646 | 0.31597931669438134 | lower_is_better | computed |
| request_latency | 198.4299857896433 | 199.88397123285813 | -1.4539854432148331 | -0.007274147267771605 | lower_is_better | computed |
| ranking_metric_value | 40216.372462257874 | 28078.79612161628 | 12137.576340641594 | 0.4322684023941311 | higher_is_better | computed |
| ttft | 190.967 | 136.758 | 54.209 | 0.3963863174366399 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 190.967 | 136.758 | 54.209 | 0.3963863174366399 | lower_is_better | computed |
| ranking_metric_value | 7743038.773604454 | 17173713.065674055 | -9430674.292069603 | -0.5491342644427404 | higher_is_better | computed |
| ttft | 494.0778564923961 | 677.828490291194 | -183.75063379879794 | -0.27108720927304036 | lower_is_better | computed |
| tpot | 173.43892824619803 | 290.114245145597 | -116.67531689939898 | -0.40217024448711297 | lower_is_better | computed |
| request_latency | 494.0778564923961 | 677.828490291194 | -183.75063379879794 | -0.27108720927304036 | lower_is_better | computed |
| ranking_metric_value | 40241.786064604836 | 28106.84994675069 | 12134.936117854144 | 0.43174301427745054 | higher_is_better | computed |
| ttft | 1908.464 | 273.243 | 1635.221 | 5.984493655830159 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 1908.464 | 273.243 | 1635.221 | 5.984493655830159 | lower_is_better | computed |
| ranking_metric_value | 7743038.773604454 | 7648177.920087521 | 94860.85351693258 | 0.012403065737760328 | higher_is_better | computed |
| ttft | 494.0778564923961 | 426.8352585556277 | 67.24259793676839 | 0.15753758994585246 | lower_is_better | computed |
| tpot | 173.43892824619803 | 164.61762927781385 | 8.821298968384184 | 0.05358659948563033 | lower_is_better | computed |
| request_latency | 494.0778564923961 | 426.8352585556277 | 67.24259793676839 | 0.15753758994585246 | lower_is_better | computed |
| ranking_metric_value | 40233.437758662236 | 28106.84994675069 | 12126.587811911544 | 0.4314459939440295 | higher_is_better | computed |
| ttft | 477.215 | 273.243 | 203.97199999999998 | 0.7464857288201344 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 477.215 | 273.243 | 203.97199999999998 | 0.7464857288201344 | lower_is_better | computed |
| ranking_metric_value | 7743038.773604454 | 17173713.065674055 | -9430674.292069603 | -0.5491342644427404 | higher_is_better | computed |
| ttft | 494.0778564923961 | 677.828490291194 | -183.75063379879794 | -0.27108720927304036 | lower_is_better | computed |
| tpot | 173.43892824619803 | 290.114245145597 | -116.67531689939898 | -0.40217024448711297 | lower_is_better | computed |
| request_latency | 494.0778564923961 | 677.828490291194 | -183.75063379879794 | -0.27108720927304036 | lower_is_better | computed |
| ranking_metric_value | 40243.49247732071 | 28106.84994675069 | 12136.642530570018 | 0.43180372590892496 | higher_is_better | computed |
| ttft | 4961.796 | 273.243 | 4688.553 | 17.158913494581746 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 4961.796 | 273.243 | 4688.553 | 17.158913494581746 | lower_is_better | computed |
| ranking_metric_value | 42652.277205623905 | 946034.6902756813 | -903382.4130700574 | -0.9549146795101195 | higher_is_better | computed |
| ttft | 1920.6477442005948 | 1775.1568914567206 | 145.49085274387426 | 0.08195943324450734 | lower_is_better | computed |
| tpot | 1283.5170762463338 | 856.9223225215032 | 426.5947537248305 | 0.49782196415372965 | lower_is_better | computed |
| request_latency | 1920.6477442005948 | 1775.1568914567206 | 145.49085274387426 | 0.08195943324450734 | lower_is_better | computed |
| ranking_metric_value | 5603.186812499609 | 11209.133692532274 | -5605.946880032665 | -0.5001231168977356 | higher_is_better | computed |
| ttft | 1279.272 | 1278.957 | 0.3149999999998272 | 0.0002462944414861697 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 1279.272 | 1278.957 | 0.3149999999998272 | 0.0002462944414861697 | lower_is_better | computed |
| ranking_metric_value | 241577.64097795196 | 1259391.3052697089 | -1017813.6642917569 | -0.8081790465226246 | higher_is_better | computed |
| ttft | 4679.638378053277 | 2328.693224836844 | 2350.945153216433 | 1.0095555430583383 | lower_is_better | computed |
| tpot | 2579.9097372596098 | 1115.546612418422 | 1464.3631248411878 | 1.312686631414314 | lower_is_better | computed |
| request_latency | 4679.638378053277 | 2328.693224836844 | 2350.945153216433 | 1.0095555430583383 | lower_is_better | computed |
| ranking_metric_value | 7614.337758340946 | 11209.133692532274 | -3594.7959341913283 | -0.3207023872492703 | higher_is_better | computed |
| ttft | 4034.494 | 1278.957 | 2755.5370000000003 | 2.154518877491581 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 4034.494 | 1278.957 | 2755.5370000000003 | 2.154518877491581 | lower_is_better | computed |
| ranking_metric_value | 200.38637645713789 | 90.89040550002517 | 109.49597095711272 | 1.2047032946407352 | higher_is_better | computed |
| ttft | 948.4342626723804 | 995.5337333705638 | -47.09947069818338 | -0.04731077322585484 | lower_is_better | computed |
| tpot | 93.88971433909059 | 16.90552351935205 | 76.98419081973853 | 4.553789223481507 | lower_is_better | computed |
| request_latency | 96997.61203156205 | 18289.884293667714 | 78707.72773789434 | 4.303347493846331 | lower_is_better | computed |
| ranking_metric_value | 690.72128 | 184.03199999999998 | 506.68928 | 2.753267257868197 | higher_is_better | computed |
| ttft | 124.503 | 938.959 | -814.4559999999999 | -0.8674031560483471 | lower_is_better | computed |
| tpot | 70.663 | 40.157 | 30.506 | 0.7596683019149838 | lower_is_better | computed |
| request_latency | 72412.752 | 42019.57 | 30393.181999999993 | 0.7233101623838605 | lower_is_better | computed |
| ranking_metric_value | 278.97553734965277 | 204.46675246225885 | 74.50878488739392 | 0.3644053812668003 | higher_is_better | computed |
| ttft | 1995.1576831814016 | 1998.6991583970541 | -3.5414752156525537 | -0.001771890081993529 | lower_is_better | computed |
| tpot | 67.94830104466277 | 220.57630377194744 | -152.62800272728467 | -0.6919510396959316 | lower_is_better | computed |
| request_latency | 71506.26965187142 | 227648.25791709928 | -156141.98826522788 | -0.685891426070516 | lower_is_better | computed |
| ranking_metric_value | 690.72128 | 184.07520000000002 | 506.64608 | 2.7523864159865092 | higher_is_better | computed |
| ttft | 124.503 | 1877.68 | -1753.1770000000001 | -0.9336931745558349 | lower_is_better | computed |
| tpot | 70.663 | 40.157 | 30.506 | 0.7596683019149838 | lower_is_better | computed |
| request_latency | 72412.752 | 42958.291 | 29454.460999999996 | 0.6856525321270345 | lower_is_better | computed |
| ranking_metric_value | 382.0630632084591 | 209.09369382095153 | 172.96936938750756 | 0.8272337927877562 | higher_is_better | computed |
| ttft | 3836.2744820888493 | 2088.5551334987285 | 1747.7193485901207 | 0.8368078584846153 | lower_is_better | computed |
| tpot | 260.6042403927421 | 296.86746415446385 | -36.26322376172175 | -0.12215290707254312 | lower_is_better | computed |
| request_latency | 270434.41240386403 | 305783.9709635153 | -35349.558559651254 | -0.11560304632144697 | lower_is_better | computed |
| ranking_metric_value | 690.72128 | 184.07520000000002 | 506.64608 | 2.7523864159865092 | higher_is_better | computed |
| ttft | 124.503 | 1877.68 | -1753.1770000000001 | -0.9336931745558349 | lower_is_better | computed |
| tpot | 70.663 | 40.157 | 30.506 | 0.7596683019149838 | lower_is_better | computed |
| request_latency | 72412.752 | 42958.291 | 29454.460999999996 | 0.6856525321270345 | lower_is_better | computed |
| ranking_metric_value | 28220.633705692715 | 2463.76529592741 | 25756.868409765306 | 10.454270320448648 | higher_is_better | computed |
| ttft | 4789.686915242221 | 4987.488061590929 | -197.80114634870733 | -0.039659472645556956 | lower_is_better | computed |
| tpot | 2654.712449529016 | 3720.6894516640036 | -1065.9770021349877 | -0.28649985869104205 | lower_is_better | computed |
| request_latency | 4789.686915242221 | 4987.488061590929 | -197.80114634870733 | -0.039659472645556956 | lower_is_better | computed |
| ranking_metric_value | 5545.37802643957 | 275.40540940331965 | 5269.97261703625 | 19.135327183492596 | higher_is_better | computed |
| ttft | 646.304 | 3718.155 | -3071.851 | -0.826176154571286 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 646.304 | 3718.155 | -3071.851 | -0.826176154571286 | lower_is_better | computed |
| ranking_metric_value | 99942.2115263856 | 5318.5408478988975 | 94623.6706784867 | 17.791283997728062 | higher_is_better | computed |
| ttft | 978.4854518071384 | 962.6700530132562 | 15.815398793882196 | 0.016428680568569026 | lower_is_better | computed |
| tpot | 428.9144843139155 | 631.41633414045 | -202.50184982653445 | -0.3207105025279417 | lower_is_better | computed |
| request_latency | 978.4854518071384 | 962.6700530132562 | 15.815398793882196 | 0.016428680568569026 | lower_is_better | computed |
| ranking_metric_value | 7885.085440092931 | 817.9270873382118 | 7067.158352754719 | 8.640328046541951 | higher_is_better | computed |
| ttft | 454.529 | 938.959 | -484.42999999999995 | -0.5159224204677734 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 454.529 | 938.959 | -484.42999999999995 | -0.5159224204677734 | lower_is_better | computed |
| ranking_metric_value | 128781.24610372537 | 145221.50251240918 | -16440.25640868381 | -0.11320814152352535 | higher_is_better | computed |
| ttft | 1447.1672362130435 | 1487.8237469106018 | -40.65651069755836 | -0.027326160630235772 | lower_is_better | computed |
| tpot | 672.9496002696623 | 695.111873455301 | -22.162273185638696 | -0.03188303067745517 | lower_is_better | computed |
| request_latency | 1447.1672362130435 | 1487.8237469106018 | -40.65651069755836 | -0.027326160630235772 | lower_is_better | computed |
| ranking_metric_value | 7885.085440092931 | 818.0307613650888 | 7067.054678727843 | 8.639106268980271 | higher_is_better | computed |
| ttft | 454.529 | 1877.68 | -1423.151 | -0.7579305312939372 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 454.529 | 1877.68 | -1423.151 | -0.7579305312939372 | lower_is_better | computed |
| ranking_metric_value | 140100.35009451074 | 230889.40823872894 | -90789.0581442182 | -0.3932144780342047 | higher_is_better | computed |
| ttft | 2382.74922064652 | 3774.205177480415 | -1391.455956833895 | -0.3686752286643843 | lower_is_better | computed |
| tpot | 1168.3265996262676 | 1838.3025887402075 | -669.9759891139399 | -0.3644535960606332 | lower_is_better | computed |
| request_latency | 2382.74922064652 | 3774.205177480415 | -1391.455956833895 | -0.3686752286643843 | lower_is_better | computed |
| ranking_metric_value | 7885.085440092931 | 818.0307613650888 | 7067.054678727843 | 8.639106268980271 | higher_is_better | computed |
| ttft | 454.529 | 1877.68 | -1423.151 | -0.7579305312939372 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 454.529 | 1877.68 | -1423.151 | -0.7579305312939372 | lower_is_better | computed |
| ranking_metric_value | 2266.4643724199677 | 1310.911657779627 | 955.5527146403408 | 0.7289222801319962 | higher_is_better | computed |
| ttft | 740.0504923149001 | 720.2390901226229 | 19.811402192277228 | 0.027506702238147444 | lower_is_better | computed |
| tpot | 91.0494843644669 | 122.49266142144401 | -31.443177056977106 | -0.2566943741127054 | lower_is_better | computed |
| request_latency | 93883.67299716454 | 126030.23172425984 | -32146.558727095296 | -0.2550702183697353 | lower_is_better | computed |
| ranking_metric_value | 1971.78816 | 1200.0096 | 771.77856 | 0.6431436548507611 | higher_is_better | computed |
| ttft | 279.399 | 288.001 | -8.601999999999975 | -0.029867951847389336 | lower_is_better | computed |
| tpot | 37.305 | 33.362 | 3.942999999999998 | 0.11818835801210952 | lower_is_better | computed |
| request_latency | 38442.414 | 34417.327 | 4025.0869999999995 | 0.11694943654398263 | lower_is_better | computed |
| ranking_metric_value | 198.2361728390824 | 161.38673657112642 | 36.84943626795598 | 0.2283300167713329 | higher_is_better | computed |
| ttft | 196.9689232997745 | 196.63706804953333 | 0.33185525024117624 | 0.0016876535717954314 | lower_is_better | computed |
| tpot | 2.960264523613805 | 2.905931932348802 | 0.05433259126500323 | 0.018697131429739777 | lower_is_better | computed |
| request_latency | 3225.3195309566972 | 3169.405434842358 | 55.91409611433937 | 0.01764182502486321 | lower_is_better | computed |
| ranking_metric_value | 1687.68 | 976.6656 | 711.0144 | 0.728001887237556 | higher_is_better | computed |
| ttft | 170.65 | 176.93 | -6.280000000000001 | -0.035494263267959084 | lower_is_better | computed |
| tpot | 32.985 | 31.093 | 1.8919999999999995 | 0.060849708937703 | lower_is_better | computed |
| request_latency | 33914.305 | 31985.069 | 1929.2360000000008 | 0.06031676842716834 | lower_is_better | computed |
| ranking_metric_value | 2266.4643724199677 | 1310.911657779627 | 955.5527146403408 | 0.7289222801319962 | higher_is_better | computed |
| ttft | 740.0504923149001 | 720.2390901226229 | 19.811402192277228 | 0.027506702238147444 | lower_is_better | computed |
| tpot | 91.0494843644669 | 122.49266142144401 | -31.443177056977106 | -0.2566943741127054 | lower_is_better | computed |
| request_latency | 93883.67299716454 | 126030.23172425984 | -32146.558727095296 | -0.2550702183697353 | lower_is_better | computed |
| ranking_metric_value | 1971.78816 | 1200.0096 | 771.77856 | 0.6431436548507611 | higher_is_better | computed |
| ttft | 279.399 | 288.001 | -8.601999999999975 | -0.029867951847389336 | lower_is_better | computed |
| tpot | 37.305 | 33.362 | 3.942999999999998 | 0.11818835801210952 | lower_is_better | computed |
| request_latency | 38442.414 | 34417.327 | 4025.0869999999995 | 0.11694943654398263 | lower_is_better | computed |
| ranking_metric_value | 1517.175389970408 | 1091.1324851403099 | 426.0429048300982 | 0.3904593719206453 | higher_is_better | computed |
| ttft | 499.64708862707 | 499.962818936633 | -0.3157303095630368 | -0.0006315075793727244 | lower_is_better | computed |
| tpot | 55.207190609278776 | 125.29801942877575 | -70.09082881949698 | -0.5593929508146721 | lower_is_better | computed |
| request_latency | 56976.60308191925 | 128679.83669457422 | -71703.23361265496 | -0.557221981737861 | lower_is_better | computed |
| ranking_metric_value | 1971.78816 | 1200.0096 | 771.77856 | 0.6431436548507611 | higher_is_better | computed |
| ttft | 279.399 | 288.001 | -8.601999999999975 | -0.029867951847389336 | lower_is_better | computed |
| tpot | 37.305 | 33.362 | 3.942999999999998 | 0.11818835801210952 | lower_is_better | computed |
| request_latency | 38442.414 | 34417.327 | 4025.0869999999995 | 0.11694943654398263 | lower_is_better | computed |
| ranking_metric_value | 2266.4643724199677 | 1310.911657779627 | 955.5527146403408 | 0.7289222801319962 | higher_is_better | computed |
| ttft | 740.0504923149001 | 720.2390901226229 | 19.811402192277228 | 0.027506702238147444 | lower_is_better | computed |
| tpot | 91.0494843644669 | 122.49266142144401 | -31.443177056977106 | -0.2566943741127054 | lower_is_better | computed |
| request_latency | 93883.67299716454 | 126030.23172425984 | -32146.558727095296 | -0.2550702183697353 | lower_is_better | computed |
| ranking_metric_value | 1971.78816 | 1200.0096 | 771.77856 | 0.6431436548507611 | higher_is_better | computed |
| ttft | 279.399 | 288.001 | -8.601999999999975 | -0.029867951847389336 | lower_is_better | computed |
| tpot | 37.305 | 33.362 | 3.942999999999998 | 0.11818835801210952 | lower_is_better | computed |
| request_latency | 38442.414 | 34417.327 | 4025.0869999999995 | 0.11694943654398263 | lower_is_better | computed |
| ranking_metric_value | 194054.80342102124 | 256791.23016936288 | -62736.42674834165 | -0.24430907047318073 | higher_is_better | computed |
| ttft | 823.1901358989784 | 933.1159784622115 | -109.92584256323312 | -0.11780512294344427 | lower_is_better | computed |
| tpot | 387.39654642530087 | 453.2003186516883 | -65.80377222638742 | -0.14519798313946372 | lower_is_better | computed |
| request_latency | 823.1901358989784 | 933.1159784622115 | -109.92584256323312 | -0.11780512294344427 | lower_is_better | computed |
| ranking_metric_value | 5655.28928919136 | 3990.0071695441325 | 1665.2821196472273 | 0.4173631898103806 | higher_is_better | computed |
| ttft | 633.743 | 449.122 | 184.62100000000004 | 0.4110709339555845 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 633.743 | 449.122 | 184.62100000000004 | 0.4110709339555845 | lower_is_better | computed |
| ranking_metric_value | 241651.5541363179 | 256791.23016936288 | -15139.67603304499 | -0.05895713815094012 | higher_is_better | computed |
| ttft | 1288.2019365139072 | 933.1159784622115 | 355.0859580516957 | 0.3805378605099898 | lower_is_better | computed |
| tpot | 649.7062831339805 | 453.2003186516883 | 196.5059644822922 | 0.43359626283343117 | lower_is_better | computed |
| request_latency | 1288.2019365139072 | 933.1159784622115 | 355.0859580516957 | 0.3805378605099898 | lower_is_better | computed |
| ranking_metric_value | 7609.188349704199 | 3990.0071695441325 | 3619.1811801600666 | 0.9070613225423267 | higher_is_better | computed |
| ttft | 1884.038 | 449.122 | 1434.916 | 3.19493589715044 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 1884.038 | 449.122 | 1434.916 | 3.19493589715044 | lower_is_better | computed |
| ranking_metric_value | 4596.783293444625 | 4113.0963240496785 | 483.6869693949466 | 0.11759680087402313 | higher_is_better | computed |
| ttft | 445.52894258047996 | 497.9217209247308 | -52.392778344250814 | -0.10522292188207404 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 445.52894258047996 | 497.9217209247308 | -52.392778344250814 | -0.10522292188207404 | lower_is_better | computed |
| ranking_metric_value | 4818.123889236925 | 3990.0071695441325 | 828.1167196927922 | 0.20754767711041644 | higher_is_better | computed |
| ttft | 371.929 | 449.122 | -77.19300000000004 | -0.17187534790101586 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 371.929 | 449.122 | -77.19300000000004 | -0.17187534790101586 | lower_is_better | computed |
| ranking_metric_value | 255979.90121312992 | 256791.23016936288 | -811.3289562329592 | -0.0031594885685849125 | higher_is_better | computed |
| ttft | 2016.1582903741194 | 933.1159784622115 | 1083.0423119119077 | 1.160672774778519 | lower_is_better | computed |
| tpot | 1080.3631753296231 | 453.2003186516883 | 627.1628566779348 | 1.3838535209856 | lower_is_better | computed |
| request_latency | 2016.1582903741194 | 933.1159784622115 | 1083.0423119119077 | 1.160672774778519 | lower_is_better | computed |
| ranking_metric_value | 7609.630620270355 | 3990.0071695441325 | 3619.623450726223 | 0.9071721670965752 | higher_is_better | computed |
| ttft | 3767.857 | 449.122 | 3318.735 | 7.389384176237192 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 3767.857 | 449.122 | 3318.735 | 7.389384176237192 | lower_is_better | computed |
| ranking_metric_value | 1392281.3918962714 | 1368381.9924646383 | 23899.399431633065 | 0.01746544427158608 | higher_is_better | computed |
| ttft | 761.9609126249366 | 511.85707197042063 | 250.10384065451592 | 0.48862046526333625 | lower_is_better | computed |
| tpot | 307.38045631246825 | 207.1285359852103 | 100.25192032725795 | 0.4840082504827644 | lower_is_better | computed |
| request_latency | 761.9609126249366 | 511.85707197042063 | 250.10384065451592 | 0.48862046526333625 | lower_is_better | computed |
| ranking_metric_value | 12834.255238386264 | 6222.200617358969 | 6612.054621027295 | 1.062655325284867 | higher_is_better | computed |
| ttft | 837.758 | 288.001 | 549.7570000000001 | 1.908871844194986 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 837.758 | 288.001 | 549.7570000000001 | 1.908871844194986 | lower_is_better | computed |
| ranking_metric_value | 12998.899340041085 | 10415.139610309046 | 2583.759729732039 | 0.24807730154424418 | higher_is_better | computed |
| ttft | 196.93975105371564 | 196.63682644953332 | 0.3029246041823228 | 0.0015405283417756347 | lower_is_better | computed |
| tpot | 79.03795619832961 | 108.50946115962667 | -29.471504961297057 | -0.271603090148443 | lower_is_better | computed |
| request_latency | 196.93975105371564 | 196.63682644953332 | 0.3029246041823228 | 0.0015405283417756347 | lower_is_better | computed |
| ranking_metric_value | 10501.025490770584 | 5064.149663708811 | 5436.875827061773 | 1.0736009375915618 | higher_is_better | computed |
| ttft | 170.65 | 176.93 | -6.280000000000001 | -0.035494263267959084 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 170.65 | 176.93 | -6.280000000000001 | -0.035494263267959084 | lower_is_better | computed |
| ranking_metric_value | 1392281.3918962714 | 1368381.9924646383 | 23899.399431633065 | 0.01746544427158608 | higher_is_better | computed |
| ttft | 761.9609126249366 | 511.85707197042063 | 250.10384065451592 | 0.48862046526333625 | lower_is_better | computed |
| tpot | 307.38045631246825 | 207.1285359852103 | 100.25192032725795 | 0.4840082504827644 | lower_is_better | computed |
| request_latency | 761.9609126249366 | 511.85707197042063 | 250.10384065451592 | 0.48862046526333625 | lower_is_better | computed |
| ranking_metric_value | 12836.183629875031 | 6222.200617358969 | 6613.983012516062 | 1.062965246421673 | higher_is_better | computed |
| ttft | 1954.475 | 288.001 | 1666.474 | 5.786347964069569 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 1954.475 | 288.001 | 1666.474 | 5.786347964069569 | lower_is_better | computed |
| ranking_metric_value | 772848.740472446 | 1191767.725885426 | -418918.98541297996 | -0.351510597504848 | higher_is_better | computed |
| ttft | 417.3649811511857 | 358.2980061678996 | 59.066974983286116 | 0.16485432228614508 | lower_is_better | computed |
| tpot | 135.08249057559286 | 130.34900308394978 | 4.733487491643075 | 0.036313952386690114 | lower_is_better | computed |
| request_latency | 417.3649811511857 | 358.2980061678996 | 59.066974983286116 | 0.16485432228614508 | lower_is_better | computed |
| ranking_metric_value | 12827.533384156708 | 6222.200617358969 | 6605.332766797739 | 1.0615750235326535 | higher_is_better | computed |
| ttft | 279.399 | 288.001 | -8.601999999999975 | -0.029867951847389336 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 279.399 | 288.001 | -8.601999999999975 | -0.029867951847389336 | lower_is_better | computed |
| ranking_metric_value | 1392281.3918962714 | 1368381.9924646383 | 23899.399431633065 | 0.01746544427158608 | higher_is_better | computed |
| ttft | 761.9609126249366 | 511.85707197042063 | 250.10384065451592 | 0.48862046526333625 | lower_is_better | computed |
| tpot | 307.38045631246825 | 207.1285359852103 | 100.25192032725795 | 0.4840082504827644 | lower_is_better | computed |
| request_latency | 761.9609126249366 | 511.85707197042063 | 250.10384065451592 | 0.48862046526333625 | lower_is_better | computed |
| ranking_metric_value | 12837.03051807641 | 6222.200617358969 | 6614.829900717441 | 1.0631013539266314 | higher_is_better | computed |
| ttft | 4746.269 | 288.001 | 4458.268 | 15.480043472071278 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 4746.269 | 288.001 | 4458.268 | 15.480043472071278 | lower_is_better | computed |
| ranking_metric_value | 188.4947719565686 | 90.89231278046095 | 97.60245917610764 | 1.0738252354943834 | higher_is_better | computed |
| ttft | 999.284585151032 | 995.4948873151652 | 3.789697835866832 | 0.0038068481156016687 | lower_is_better | computed |
| tpot | 93.85339854460317 | 16.905186326682674 | 76.94821221792049 | 4.551751795628989 | lower_is_better | computed |
| request_latency | 97011.31129628008 | 18289.50049951154 | 78721.81079676854 | 4.304207804847977 | lower_is_better | computed |
| ranking_metric_value | 631.1232000000001 | 184.03199999999998 | 447.09120000000013 | 2.4294209702660416 | higher_is_better | computed |
| ttft | 730.137 | 938.912 | -208.7750000000001 | -0.2223584318871205 | lower_is_better | computed |
| tpot | 57.03 | 40.157 | 16.873000000000005 | 0.42017580994596226 | lower_is_better | computed |
| request_latency | 59071.827000000005 | 42019.522999999994 | 17052.30400000001 | 0.4058186000826333 | lower_is_better | computed |
| ranking_metric_value | 269.44971807606487 | 204.47479049635606 | 64.9749275797088 | 0.317764979350189 | higher_is_better | computed |
| ttft | 1997.9554134072716 | 1998.6047562815402 | -0.649342874268541 | -0.0003248980931460713 | lower_is_better | computed |
| tpot | 62.99414685160754 | 220.56764825315125 | -157.5735014015437 | -0.7143998798078152 | lower_is_better | computed |
| request_latency | 66440.96764260178 | 227639.30891925527 | -161198.3412766535 | -0.708130515955095 | lower_is_better | computed |
| ranking_metric_value | 631.1232000000001 | 184.07520000000002 | 447.0480000000001 | 2.4286161307987175 | higher_is_better | computed |
| ttft | 730.137 | 1877.585 | -1147.448 | -0.6111297224892616 | lower_is_better | computed |
| tpot | 57.03 | 40.157 | 16.873000000000005 | 0.42017580994596226 | lower_is_better | computed |
| request_latency | 59071.827000000005 | 42958.195999999996 | 16113.631000000008 | 0.37510027190154843 | lower_is_better | computed |
| ranking_metric_value | 382.07633435799437 | 209.10215789861894 | 172.97417645937543 | 0.8272232969649228 | higher_is_better | computed |
| ttft | 3836.106949745501 | 2088.4570120611365 | 1747.6499376843644 | 0.836813938516061 | lower_is_better | computed |
| tpot | 260.5952220012536 | 296.8554607708919 | -36.260238769638306 | -0.1221477909669425 | lower_is_better | computed |
| request_latency | 270425.01905702794 | 305771.59338068357 | -35346.57432365563 | -0.11559796622326975 | lower_is_better | computed |
| ranking_metric_value | 631.1232000000001 | 184.07520000000002 | 447.0480000000001 | 2.4286161307987175 | higher_is_better | computed |
| ttft | 730.137 | 1877.585 | -1147.448 | -0.6111297224892616 | lower_is_better | computed |
| tpot | 57.03 | 40.157 | 16.873000000000005 | 0.42017580994596226 | lower_is_better | computed |
| request_latency | 59071.827000000005 | 42958.195999999996 | 16113.631000000008 | 0.37510027190154843 | lower_is_better | computed |
| ranking_metric_value | 26657.745364540093 | 2463.8152397109134 | 24193.93012482918 | 9.819701467415197 | higher_is_better | computed |
| ttft | 4993.6706266642905 | 4987.386960656103 | 6.283666008187538 | 0.001259911464211092 | lower_is_better | computed |
| tpot | 2775.374263547458 | 3720.613040672604 | -945.2387771251456 | -0.25405457831601524 | lower_is_better | computed |
| request_latency | 4993.6706266642905 | 4987.386960656103 | 6.283666008187538 | 0.001259911464211092 | lower_is_better | computed |
| ranking_metric_value | 3887.9776441285458 | 275.411038872493 | 3612.5666052560528 | 13.117000030374825 | higher_is_better | computed |
| ttft | 921.816 | 3718.079 | -2796.263 | -0.7520719704987441 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 921.816 | 3718.079 | -2796.263 | -0.7520719704987441 | lower_is_better | computed |
| ranking_metric_value | 74310.76900320637 | 5318.748446563601 | 68992.02055664276 | 12.971476513655752 | higher_is_better | computed |
| ttft | 999.0476615414474 | 962.6324785688989 | 36.41518297254845 | 0.03782874958331472 | lower_is_better | computed |
| tpot | 452.91048467667184 | 631.3897842849537 | -178.4792996082819 | -0.2826768884301937 | lower_is_better | computed |
| request_latency | 999.0476615414474 | 962.6324785688989 | 36.41518297254845 | 0.03782874958331472 | lower_is_better | computed |
| ranking_metric_value | 4908.667825353325 | 817.9680310827852 | 4090.6997942705393 | 5.001050944320496 | higher_is_better | computed |
| ttft | 730.137 | 938.912 | -208.7750000000001 | -0.2223584318871205 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 730.137 | 938.912 | -208.7750000000001 | -0.2223584318871205 | lower_is_better | computed |
| ranking_metric_value | 124288.97008960613 | 145226.96848683534 | -20937.99839722921 | -0.14417431290750393 | higher_is_better | computed |
| ttft | 1499.4733632890996 | 1487.767749001701 | 11.705614287398475 | 0.00786790431184773 | lower_is_better | computed |
| tpot | 699.9327417360679 | 695.0838745008506 | 4.8488672352173126 | 0.006975945512617958 | lower_is_better | computed |
| request_latency | 1499.4733632890996 | 1487.767749001701 | 11.705614287398475 | 0.00786790431184773 | lower_is_better | computed |
| ranking_metric_value | 4908.667825353325 | 818.0721511942202 | 4090.5956741591044 | 5.000287160720067 | higher_is_better | computed |
| ttft | 730.137 | 1877.585 | -1147.448 | -0.6111297224892616 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 730.137 | 1877.585 | -1147.448 | -0.6111297224892616 | lower_is_better | computed |
| ranking_metric_value | 138112.15631820148 | 230896.1272202227 | -92783.97090202122 | -0.40184290667433453 | higher_is_better | computed |
| ttft | 2417.050091020888 | 3774.0953496758243 | -1357.0452586549363 | -0.359568355572018 | lower_is_better | computed |
| tpot | 1186.2047560596593 | 1838.2476748379122 | -652.0429187782529 | -0.35470895881086695 | lower_is_better | computed |
| request_latency | 2417.050091020888 | 3774.0953496758243 | -1357.0452586549363 | -0.359568355572018 | lower_is_better | computed |
| ranking_metric_value | 4908.667825353325 | 818.0721511942202 | 4090.5956741591044 | 5.000287160720067 | higher_is_better | computed |
| ttft | 730.137 | 1877.585 | -1147.448 | -0.6111297224892616 | lower_is_better | computed |
| tpot | 0.0 | 0.0 | 0.0 | N/A (zero baseline, both zero) | lower_is_better | zero_baseline_both_zero |
| request_latency | 730.137 | 1877.585 | -1147.448 | -0.6111297224892616 | lower_is_better | computed |
