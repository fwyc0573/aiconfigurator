## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-07-15 | Rebound the numerically identical comparison tables to the formatter-consistent corrected-v3 merged source identity and hashes. |
| 2026-07-15 | Rebound the numerically identical comparison tables to the corrected-v2 merged source identity and hashes. |
| 2026-07-14 | Added the final merged Step4 versus DeepSeek-V4-Pro numeric comparison summary. |

# Step4 vs DeepSeek-V4-Pro Comparison Summary

## Scope and Interpretation

- Source JSON: `task_memory/task_2026-07-10_step4_predefined_ops_plan/result-full-matrix-v3-merged-20260715/results.json`
- Raw per-key comparison CSV: `task_memory/task_2026-07-10_step4_predefined_ops_plan/result-full-matrix-v3-merged-20260715/model_comparisons.csv`
- Source JSON SHA256: `372b683274771b70f3a5a94caafe08ac0289268140c038194995a5fa72c24a3e`
- Matrix identity: `78b4970381ca7d0fb7bdfb53619a3280b72f936703c3c4d813bd23b96398edd4`
- Execution contract: `a13a4fe6ef9b932d01772ee3f0b8844760c52ec991fdcc5af641186baf1b697c`
- Paired rank-one comparisons: `89`; unpaired comparisons: `28`.
- Every raw comparison key preserves `system`, `workload`, `ISL`, `OSL`, `TTFT SLA`, and `serving mode`.
- The aggregate tables below use paired points only. Means do not impute missing/unpaired points.
- Absolute delta is `Step4 - DeepSeek-V4-Pro`. Relative delta uses DeepSeek as baseline.
- `Relative n` excludes legal TPOT `0/0` rows whose relative delta is undefined.
- A win respects metric polarity: throughput is higher-is-better; TTFT, TPOT, and request latency are lower-is-better.
- `ranking_metric_value` means Prefill input throughput for primary `OSL=1`, and output-token throughput for decode smoke `OSL=1024`; do not combine those meanings outside the workload-specific tables.
- H800 values are **simulated SOL validation**, not silicon measurements.
- Step4 uses the declared temporary MLA substitute for all 92 attention layers; Step4 rows at ISL >= 65536 are approximation dominated.

## Comparison Coverage

| System | Paired | Unpaired | Step4-only | DeepSeek-only | Validation type |
|---|---:|---:|---:|---:|---|
| gb300 | 33 | 2 | 0 | 2 | SOL model |
| h200_sxm | 28 | 2 | 2 | 0 | SOL model |
| h100_sxm | 14 | 14 | 14 | 0 | SOL model |
| h800_sxm | 14 | 10 | 10 | 0 | simulated SOL |

| Workload | Mode | Paired | Unpaired | Step4-only | DeepSeek-only |
|---|---|---:|---:|---:|---:|
| primary | agg | 28 | 10 | 9 | 1 |
| primary | disagg | 29 | 11 | 10 | 1 |
| decode_smoke | agg | 16 | 3 | 3 | 0 |
| decode_smoke | disagg | 16 | 4 | 4 | 0 |

## Overall by Metric

| Metric | Pairs | Step4 mean | DeepSeek mean | Mean absolute delta | Mean relative delta | Relative n | Step4 wins | Ties | DeepSeek wins |
|---|---|---|---|---|---|---|---|---|---|
| ranking_metric_value (tokens/s/gpu_cluster) | 89 | 500093.116774 | 1.116074e+06 | -615981.193155 | 170.058% | 89 | 67 | 0 | 22 |
| request_latency (ms) | 89 | 27750.140615 | 32823.907914 | -5073.767299 | 84.314% | 89 | 41 | 0 | 48 |
| tpot (ms/token) | 89 | 252.678797 | 266.397176 | -13.718379 | 17.863% | 59 | 30 | 30 | 29 |
| ttft (ms) | 89 | 1172.621609 | 1033.896741 | 138.724868 | 69.608% | 89 | 49 | 0 | 40 |

## By System, Workload, Serving Mode, and Metric

| System | Workload | Mode | Metric | Pairs | Step4 mean | DeepSeek mean | Mean absolute delta | Mean relative delta | Relative n | Step4 wins | Ties | DeepSeek wins |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gb300 | decode_smoke | agg | ranking_metric_value (tokens/s/gpu_cluster) | 5 | 4967.241037 | 4737.578806 | 229.662231 | -3.205% | 5 | 4 | 0 | 1 |
| gb300 | decode_smoke | agg | request_latency (ms) | 5 | 67122.567609 | 108919.775132 | -41797.207523 | -38.023% | 5 | 5 | 0 | 0 |
| gb300 | decode_smoke | agg | tpot (ms/token) | 5 | 65.270648 | 106.101875 | -40.831226 | -38.170% | 5 | 5 | 0 | 0 |
| gb300 | decode_smoke | agg | ttft (ms) | 5 | 350.694266 | 377.557139 | -26.862873 | -6.374% | 5 | 5 | 0 | 0 |
| gb300 | decode_smoke | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 5 | 5206.817280 | 5058.236160 | 148.581120 | 2.937% | 5 | 5 | 0 | 0 |
| gb300 | decode_smoke | disagg | request_latency (ms) | 5 | 51119.827000 | 54536.146800 | -3416.319800 | -5.715% | 5 | 4 | 0 | 1 |
| gb300 | decode_smoke | disagg | tpot (ms/token) | 5 | 49.876000 | 53.069600 | -3.193600 | -5.478% | 5 | 4 | 0 | 1 |
| gb300 | decode_smoke | disagg | ttft (ms) | 5 | 96.679000 | 245.946000 | -149.267000 | -57.556% | 5 | 5 | 0 | 0 |
| gb300 | primary | agg | ranking_metric_value (tokens/s/gpu_cluster) | 11 | 3.415231e+06 | 8.373979e+06 | -4.958748e+06 | -66.379% | 11 | 1 | 0 | 10 |
| gb300 | primary | agg | request_latency (ms) | 11 | 1251.532242 | 994.010646 | 257.521596 | 9.412% | 11 | 5 | 0 | 6 |
| gb300 | primary | agg | tpot (ms/token) | 11 | 625.740144 | 450.268102 | 175.472043 | 13.669% | 11 | 4 | 0 | 7 |
| gb300 | primary | agg | ttft (ms) | 11 | 1251.532242 | 994.010646 | 257.521596 | 9.412% | 11 | 5 | 0 | 6 |
| gb300 | primary | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 12 | 25948.602167 | 21150.200915 | 4798.401252 | 14.339% | 12 | 9 | 0 | 3 |
| gb300 | primary | disagg | request_latency (ms) | 12 | 1769.051833 | 538.606500 | 1230.445333 | 296.977% | 12 | 2 | 0 | 10 |
| gb300 | primary | disagg | tpot (ms/token) | 12 | 0 | 0 | 0 | N/A | 0 | 0 | 12 | 0 |
| gb300 | primary | disagg | ttft (ms) | 12 | 1769.051833 | 538.606500 | 1230.445333 | 296.977% | 12 | 2 | 0 | 10 |
| h100_sxm | decode_smoke | agg | ranking_metric_value (tokens/s/gpu_cluster) | 3 | 287.141659 | 168.150284 | 118.991375 | 79.878% | 3 | 3 | 0 | 0 |
| h100_sxm | decode_smoke | agg | request_latency (ms) | 3 | 146312.764696 | 183907.371058 | -37594.606362 | 116.728% | 3 | 2 | 0 | 1 |
| h100_sxm | decode_smoke | agg | tpot (ms/token) | 3 | 140.814085 | 178.116430 | -37.302345 | 124.656% | 3 | 2 | 0 | 1 |
| h100_sxm | decode_smoke | agg | ttft (ms) | 3 | 2259.955476 | 1694.262675 | 565.692801 | 26.258% | 3 | 2 | 0 | 1 |
| h100_sxm | decode_smoke | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 3 | 690.721280 | 184.060800 | 506.660480 | 275.268% | 3 | 3 | 0 | 0 |
| h100_sxm | decode_smoke | disagg | request_latency (ms) | 3 | 72412.752000 | 42645.384000 | 29767.368000 | 69.821% | 3 | 0 | 0 | 3 |
| h100_sxm | decode_smoke | disagg | tpot (ms/token) | 3 | 70.663000 | 40.157000 | 30.506000 | 75.967% | 3 | 0 | 0 | 3 |
| h100_sxm | decode_smoke | disagg | ttft (ms) | 3 | 124.503000 | 1564.773000 | -1440.270000 | -91.160% | 3 | 3 | 0 | 0 |
| h100_sxm | primary | agg | ranking_metric_value (tokens/s/gpu_cluster) | 4 | 99261.110358 | 95973.304224 | 3287.806134 | 693.478% | 4 | 2 | 0 | 2 |
| h100_sxm | primary | agg | request_latency (ms) | 4 | 2399.522206 | 2803.046760 | -403.524554 | -10.481% | 4 | 3 | 0 | 1 |
| h100_sxm | primary | agg | tpot (ms/token) | 4 | 1231.225783 | 1721.380062 | -490.154279 | -25.089% | 4 | 4 | 0 | 0 |
| h100_sxm | primary | agg | ttft (ms) | 4 | 2399.522206 | 2803.046760 | -403.524554 | -10.481% | 4 | 3 | 0 | 1 |
| h100_sxm | primary | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 4 | 7300.158587 | 682.348505 | 6617.810082 | 1126.347% | 4 | 4 | 0 | 0 |
| h100_sxm | primary | disagg | request_latency (ms) | 4 | 502.472750 | 2103.118500 | -1600.645750 | -71.449% | 4 | 4 | 0 | 0 |
| h100_sxm | primary | disagg | tpot (ms/token) | 4 | 0 | 0 | 0 | N/A | 0 | 0 | 4 | 0 |
| h100_sxm | primary | disagg | ttft (ms) | 4 | 502.472750 | 2103.118500 | -1600.645750 | -71.449% | 4 | 4 | 0 | 0 |
| h200_sxm | decode_smoke | agg | ranking_metric_value (tokens/s/gpu_cluster) | 5 | 1702.960936 | 1037.050839 | 665.910097 | 56.111% | 5 | 5 | 0 | 0 |
| h200_sxm | decode_smoke | agg | request_latency (ms) | 5 | 68370.588321 | 101987.987460 | -33617.399140 | -26.096% | 5 | 4 | 0 | 1 |
| h200_sxm | decode_smoke | agg | tpot (ms/token) | 5 | 66.263182 | 99.136387 | -32.873205 | -26.216% | 5 | 4 | 0 | 1 |
| h200_sxm | decode_smoke | agg | ttft (ms) | 5 | 583.353498 | 571.463431 | 11.890066 | 1.672% | 5 | 1 | 0 | 4 |
| h200_sxm | decode_smoke | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 5 | 1914.966528 | 1155.340800 | 759.625728 | 66.012% | 5 | 5 | 0 | 0 |
| h200_sxm | decode_smoke | disagg | request_latency (ms) | 5 | 37536.792200 | 33930.875400 | 3605.916800 | 10.562% | 5 | 0 | 0 | 5 |
| h200_sxm | decode_smoke | disagg | tpot (ms/token) | 5 | 36.441000 | 32.908200 | 3.532800 | 10.672% | 5 | 0 | 0 | 5 |
| h200_sxm | decode_smoke | disagg | ttft (ms) | 5 | 257.649200 | 265.786800 | -8.137600 | -3.099% | 5 | 5 | 0 | 0 |
| h200_sxm | primary | agg | ranking_metric_value (tokens/s/gpu_cluster) | 9 | 628774.984174 | 675757.292191 | -46982.308017 | -2.665% | 9 | 5 | 0 | 4 |
| h200_sxm | primary | agg | request_latency (ms) | 9 | 830.362975 | 598.641745 | 231.721230 | 32.783% | 9 | 2 | 0 | 7 |
| h200_sxm | primary | agg | tpot (ms/token) | 9 | 361.525313 | 246.649448 | 114.875866 | 36.112% | 8 | 2 | 1 | 6 |
| h200_sxm | primary | agg | ttft (ms) | 9 | 830.362975 | 598.641745 | 231.721230 | 32.783% | 9 | 2 | 0 | 7 |
| h200_sxm | primary | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 9 | 9725.362268 | 5101.442312 | 4623.919955 | 86.256% | 9 | 9 | 0 | 0 |
| h200_sxm | primary | disagg | request_latency (ms) | 9 | 1627.346444 | 347.269111 | 1280.077333 | 377.038% | 9 | 3 | 0 | 6 |
| h200_sxm | primary | disagg | tpot (ms/token) | 9 | 0 | 0 | 0 | N/A | 0 | 0 | 9 | 0 |
| h200_sxm | primary | disagg | ttft (ms) | 9 | 1627.346444 | 347.269111 | 1280.077333 | 377.038% | 9 | 3 | 0 | 6 |
| h800_sxm | decode_smoke | agg | ranking_metric_value (tokens/s/gpu_cluster) | 3 | 280.006941 | 168.156420 | 111.850521 | 73.960% | 3 | 3 | 0 | 0 |
| h800_sxm | decode_smoke | agg | request_latency (ms) | 3 | 144625.765999 | 183900.134266 | -39274.368268 | 116.016% | 3 | 2 | 0 | 1 |
| h800_sxm | decode_smoke | agg | tpot (ms/token) | 3 | 139.147589 | 178.109432 | -38.961843 | 123.840% | 3 | 2 | 0 | 1 |
| h800_sxm | decode_smoke | agg | ttft (ms) | 3 | 2277.782316 | 1694.185552 | 583.596764 | 28.010% | 3 | 1 | 0 | 2 |
| h800_sxm | decode_smoke | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 3 | 631.123200 | 184.060800 | 447.062400 | 242.888% | 3 | 3 | 0 | 0 |
| h800_sxm | decode_smoke | disagg | request_latency (ms) | 3 | 59071.827000 | 42645.305000 | 16426.522000 | 38.534% | 3 | 0 | 0 | 3 |
| h800_sxm | decode_smoke | disagg | tpot (ms/token) | 3 | 57.030000 | 40.157000 | 16.873000 | 42.018% | 3 | 0 | 0 | 3 |
| h800_sxm | decode_smoke | disagg | ttft (ms) | 3 | 730.137000 | 1564.694000 | -834.557000 | -48.154% | 3 | 3 | 0 | 0 |
| h800_sxm | primary | agg | ranking_metric_value (tokens/s/gpu_cluster) | 4 | 90842.410194 | 95976.414848 | -5134.004654 | 556.129% | 4 | 2 | 0 | 2 |
| h800_sxm | primary | agg | request_latency (ms) | 4 | 2477.310436 | 2802.970634 | -325.660199 | -7.815% | 4 | 1 | 0 | 3 |
| h800_sxm | primary | agg | tpot (ms/token) | 4 | 1278.605562 | 1721.333594 | -442.728032 | -22.112% | 4 | 3 | 0 | 1 |
| h800_sxm | primary | agg | ttft (ms) | 4 | 2477.310436 | 2802.970634 | -325.660199 | -7.815% | 4 | 1 | 0 | 3 |
| h800_sxm | primary | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 4 | 4653.495280 | 682.380843 | 3971.114437 | 702.966% | 4 | 4 | 0 | 0 |
| h800_sxm | primary | disagg | request_latency (ms) | 4 | 778.056750 | 2103.040250 | -1324.983500 | -54.917% | 4 | 4 | 0 | 0 |
| h800_sxm | primary | disagg | tpot (ms/token) | 4 | 0 | 0 | 0 | N/A | 0 | 0 | 4 | 0 |
| h800_sxm | primary | disagg | ttft (ms) | 4 | 778.056750 | 2103.040250 | -1324.983500 | -54.917% | 4 | 4 | 0 | 0 |

## Primary Workload by ISL, Serving Mode, and Metric

| Workload | ISL | Mode | Metric | Pairs | Step4 mean | DeepSeek mean | Mean absolute delta | Mean relative delta | Relative n | Step4 wins | Ties | DeepSeek wins |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | 16384 | agg | ranking_metric_value (tokens/s/gpu_cluster) | 10 | 699007.470635 | 3.111373e+06 | -2.412365e+06 | 170.030% | 10 | 3 | 0 | 7 |
| primary | 16384 | agg | request_latency (ms) | 10 | 1934.826397 | 1744.220697 | 190.605701 | 18.866% | 10 | 4 | 0 | 6 |
| primary | 16384 | agg | tpot (ms/token) | 10 | 980.022664 | 1069.073459 | -89.050796 | 18.463% | 9 | 4 | 1 | 5 |
| primary | 16384 | agg | ttft (ms) | 10 | 1934.826397 | 1744.220697 | 190.605701 | 18.866% | 10 | 4 | 0 | 6 |
| primary | 16384 | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 11 | 12010.654326 | 9762.617529 | 2248.036797 | 318.855% | 11 | 10 | 0 | 1 |
| primary | 16384 | disagg | request_latency (ms) | 11 | 1422.527545 | 1082.577818 | 339.949727 | 144.991% | 11 | 5 | 0 | 6 |
| primary | 16384 | disagg | tpot (ms/token) | 11 | 0 | 0 | 0 | N/A | 0 | 0 | 11 | 0 |
| primary | 16384 | disagg | ttft (ms) | 11 | 1422.527545 | 1082.577818 | 339.949727 | 144.991% | 11 | 5 | 0 | 6 |
| primary | 4096 | agg | ranking_metric_value (tokens/s/gpu_cluster) | 16 | 2.294539e+06 | 4.102765e+06 | -1.808225e+06 | 170.017% | 16 | 7 | 0 | 9 |
| primary | 4096 | agg | request_latency (ms) | 16 | 924.931369 | 1074.994081 | -150.062712 | 1.724% | 16 | 7 | 0 | 9 |
| primary | 4096 | agg | tpot (ms/token) | 16 | 407.033834 | 517.527828 | -110.493994 | -6.048% | 16 | 9 | 0 | 7 |
| primary | 4096 | agg | ttft (ms) | 16 | 924.931369 | 1074.994081 | -150.062712 | 1.724% | 16 | 7 | 0 | 9 |
| primary | 4096 | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 16 | 18836.961233 | 10960.453062 | 7876.508172 | 302.519% | 16 | 16 | 0 | 0 |
| primary | 4096 | disagg | request_latency (ms) | 16 | 1252.205563 | 746.691563 | 505.514000 | 290.077% | 16 | 8 | 0 | 8 |
| primary | 4096 | disagg | tpot (ms/token) | 16 | 0 | 0 | 0 | N/A | 0 | 0 | 16 | 0 |
| primary | 4096 | disagg | ttft (ms) | 16 | 1252.205563 | 746.691563 | 505.514000 | 290.077% | 16 | 8 | 0 | 8 |
| primary | 65536 | agg | ranking_metric_value (tokens/s/gpu_cluster) | 2 | 142114.959092 | 1.102713e+06 | -960598.038681 | -88.155% | 2 | 0 | 0 | 2 |
| primary | 65536 | agg | request_latency (ms) | 2 | 3300.143061 | 2051.925058 | 1248.218003 | 54.576% | 2 | 0 | 0 | 2 |
| primary | 65536 | agg | tpot (ms/token) | 2 | 1931.713407 | 986.234467 | 945.478939 | 90.525% | 2 | 0 | 0 | 2 |
| primary | 65536 | agg | ttft (ms) | 2 | 3300.143061 | 2051.925058 | 1248.218003 | 54.576% | 2 | 0 | 0 | 2 |
| primary | 65536 | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 2 | 6608.762285 | 11209.133693 | -4600.371407 | -41.041% | 2 | 0 | 0 | 2 |
| primary | 65536 | disagg | request_latency (ms) | 2 | 2656.883000 | 1278.957000 | 1377.926000 | 107.738% | 2 | 0 | 0 | 2 |
| primary | 65536 | disagg | tpot (ms/token) | 2 | 0 | 0 | 0 | N/A | 0 | 0 | 2 | 0 |
| primary | 65536 | disagg | ttft (ms) | 2 | 2656.883000 | 1278.957000 | 1377.926000 | 107.738% | 2 | 0 | 0 | 2 |

## By TTFT SLA, Workload, Serving Mode, and Metric

| TTFT SLA (ms) | Workload | Mode | Metric | Pairs | Step4 mean | DeepSeek mean | Mean absolute delta | Mean relative delta | Relative n | Step4 wins | Ties | DeepSeek wins |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | decode_smoke | agg | ranking_metric_value (tokens/s/gpu_cluster) | 4 | 2182.830673 | 1794.898192 | 387.932481 | 76.897% | 4 | 4 | 0 | 0 |
| 1000 | decode_smoke | agg | request_latency (ms) | 4 | 92093.122636 | 73346.895432 | 18746.227204 | 199.197% | 4 | 2 | 0 | 2 |
| 1000 | decode_smoke | agg | tpot (ms/token) | 4 | 89.270843 | 70.932146 | 18.338697 | 211.581% | 4 | 2 | 0 | 2 |
| 1000 | decode_smoke | agg | ttft (ms) | 4 | 769.049880 | 783.310060 | -14.260180 | -2.387% | 4 | 2 | 0 | 2 |
| 1000 | decode_smoke | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 4 | 2125.112480 | 1656.828000 | 468.284480 | 146.375% | 4 | 4 | 0 | 0 |
| 1000 | decode_smoke | disagg | request_latency (ms) | 4 | 55261.705000 | 43739.458750 | 11522.246250 | 28.771% | 4 | 1 | 0 | 3 |
| 1000 | decode_smoke | disagg | tpot (ms/token) | 4 | 53.718500 | 42.160000 | 11.558500 | 30.137% | 4 | 1 | 0 | 3 |
| 1000 | decode_smoke | disagg | ttft (ms) | 4 | 307.679500 | 609.778750 | -302.099250 | -44.145% | 4 | 4 | 0 | 0 |
| 1000 | primary | agg | ranking_metric_value (tokens/s/gpu_cluster) | 6 | 1.809183e+06 | 4.620645e+06 | -2.811462e+06 | 485.640% | 6 | 3 | 0 | 3 |
| 1000 | primary | agg | request_latency (ms) | 6 | 833.727470 | 827.539037 | 6.188432 | 3.084% | 6 | 2 | 0 | 4 |
| 1000 | primary | agg | tpot (ms/token) | 6 | 361.890988 | 437.169049 | -75.278061 | -10.643% | 6 | 4 | 0 | 2 |
| 1000 | primary | agg | ttft (ms) | 6 | 833.727470 | 827.539037 | 6.188432 | 3.084% | 6 | 2 | 0 | 4 |
| 1000 | primary | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 6 | 15656.169950 | 10067.472183 | 5588.697767 | 260.819% | 6 | 6 | 0 | 0 |
| 1000 | primary | disagg | request_latency (ms) | 6 | 715.954833 | 598.211333 | 117.743500 | 67.528% | 6 | 3 | 0 | 3 |
| 1000 | primary | disagg | tpot (ms/token) | 6 | 0 | 0 | 0 | N/A | 0 | 0 | 6 | 0 |
| 1000 | primary | disagg | ttft (ms) | 6 | 715.954833 | 598.211333 | 117.743500 | 67.528% | 6 | 3 | 0 | 3 |
| 200 | decode_smoke | agg | ranking_metric_value (tokens/s/gpu_cluster) | 2 | 365.266337 | 550.843600 | -185.577263 | -10.279% | 2 | 1 | 0 | 1 |
| 200 | decode_smoke | agg | request_latency (ms) | 2 | 8459.290350 | 12328.210128 | -3868.919778 | -17.254% | 2 | 1 | 0 | 1 |
| 200 | decode_smoke | agg | tpot (ms/token) | 2 | 8.075201 | 11.857228 | -3.782027 | -17.371% | 2 | 1 | 0 | 1 |
| 200 | decode_smoke | agg | ttft (ms) | 2 | 198.359771 | 198.266325 | 0.093446 | 0.048% | 2 | 1 | 0 | 1 |
| 200 | decode_smoke | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 2 | 3447.248640 | 3015.446400 | 431.802240 | 37.910% | 2 | 2 | 0 | 0 |
| 200 | decode_smoke | disagg | request_latency (ms) | 2 | 42517.066000 | 39330.071500 | 3186.994500 | 7.777% | 2 | 0 | 0 | 2 |
| 200 | decode_smoke | disagg | tpot (ms/token) | 2 | 41.430500 | 38.292500 | 3.138000 | 7.861% | 2 | 0 | 0 | 2 |
| 200 | decode_smoke | disagg | ttft (ms) | 2 | 133.664500 | 156.844000 | -23.179500 | -16.428% | 2 | 2 | 0 | 0 |
| 200 | primary | agg | ranking_metric_value (tokens/s/gpu_cluster) | 2 | 42623.022282 | 207564.966201 | -164941.943920 | -28.670% | 2 | 1 | 0 | 1 |
| 200 | primary | agg | request_latency (ms) | 2 | 197.684868 | 198.260399 | -0.575530 | -0.287% | 2 | 1 | 0 | 1 |
| 200 | primary | agg | tpot (ms/token) | 2 | 76.161550 | 82.099068 | -5.937518 | 2.219% | 2 | 1 | 0 | 1 |
| 200 | primary | agg | ttft (ms) | 2 | 197.684868 | 198.260399 | -0.575530 | -0.287% | 2 | 1 | 0 | 1 |
| 200 | primary | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 3 | 20447.420281 | 14119.560826 | 6327.859455 | 55.292% | 3 | 3 | 0 | 0 |
| 200 | primary | disagg | request_latency (ms) | 3 | 176.759333 | 169.379333 | 7.380000 | 7.609% | 3 | 2 | 0 | 1 |
| 200 | primary | disagg | tpot (ms/token) | 3 | 0 | 0 | 0 | N/A | 0 | 0 | 3 | 0 |
| 200 | primary | disagg | ttft (ms) | 3 | 176.759333 | 169.379333 | 7.380000 | 7.609% | 3 | 2 | 0 | 1 |
| 2000 | decode_smoke | agg | ranking_metric_value (tokens/s/gpu_cluster) | 4 | 2222.716700 | 1851.687898 | 371.028802 | 36.988% | 4 | 4 | 0 | 0 |
| 2000 | decode_smoke | agg | request_latency (ms) | 4 | 78077.701128 | 178023.940942 | -99946.239815 | -50.842% | 4 | 4 | 0 | 0 |
| 2000 | decode_smoke | agg | tpot (ms/token) | 4 | 75.070677 | 172.765457 | -97.694779 | -51.216% | 4 | 4 | 0 | 0 |
| 2000 | decode_smoke | agg | ttft (ms) | 4 | 1280.398442 | 1284.878883 | -4.480441 | -1.352% | 4 | 3 | 0 | 1 |
| 2000 | decode_smoke | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 4 | 2125.112480 | 1656.849600 | 468.262880 | 146.333% | 4 | 4 | 0 | 0 |
| 2000 | decode_smoke | disagg | request_latency (ms) | 4 | 55261.705000 | 44208.807250 | 11052.897750 | 27.061% | 4 | 1 | 0 | 3 |
| 2000 | decode_smoke | disagg | tpot (ms/token) | 4 | 53.718500 | 42.160000 | 11.558500 | 30.137% | 4 | 1 | 0 | 3 |
| 2000 | decode_smoke | disagg | ttft (ms) | 4 | 307.679500 | 1079.127250 | -771.447750 | -55.522% | 4 | 4 | 0 | 0 |
| 2000 | primary | agg | ranking_metric_value (tokens/s/gpu_cluster) | 7 | 1.684065e+06 | 4.250588e+06 | -2.566523e+06 | -36.932% | 7 | 1 | 0 | 6 |
| 2000 | primary | agg | request_latency (ms) | 7 | 1312.122091 | 1178.573882 | 133.548209 | 13.556% | 7 | 2 | 0 | 5 |
| 2000 | primary | agg | tpot (ms/token) | 7 | 658.427013 | 548.142113 | 110.284900 | 18.199% | 7 | 2 | 0 | 5 |
| 2000 | primary | agg | ttft (ms) | 7 | 1312.122091 | 1178.573882 | 133.548209 | 13.556% | 7 | 2 | 0 | 5 |
| 2000 | primary | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 7 | 14500.141789 | 10230.596370 | 4269.545419 | 223.389% | 7 | 6 | 0 | 1 |
| 2000 | primary | disagg | request_latency (ms) | 7 | 1368.758571 | 963.659857 | 405.098714 | 207.883% | 7 | 2 | 0 | 5 |
| 2000 | primary | disagg | tpot (ms/token) | 7 | 0 | 0 | 0 | N/A | 0 | 0 | 7 | 0 |
| 2000 | primary | disagg | ttft (ms) | 7 | 1368.758571 | 963.659857 | 405.098714 | 207.883% | 7 | 2 | 0 | 5 |
| 500 | decode_smoke | agg | ranking_metric_value (tokens/s/gpu_cluster) | 2 | 3796.576281 | 3389.015438 | 407.560842 | 22.944% | 2 | 2 | 0 | 0 |
| 500 | decode_smoke | agg | request_latency (ms) | 2 | 68728.248650 | 129728.900952 | -61000.652302 | -47.091% | 2 | 2 | 0 | 0 |
| 500 | decode_smoke | agg | tpot (ms/token) | 2 | 66.748983 | 126.361616 | -59.612633 | -47.249% | 2 | 2 | 0 | 0 |
| 500 | decode_smoke | agg | ttft (ms) | 2 | 444.038633 | 460.967674 | -16.929040 | -4.006% | 2 | 2 | 0 | 0 |
| 500 | decode_smoke | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 2 | 3589.302720 | 3129.624000 | 459.678720 | 33.616% | 2 | 2 | 0 | 0 |
| 500 | decode_smoke | disagg | request_latency (ms) | 2 | 44781.120500 | 45459.371000 | -678.250500 | 1.085% | 2 | 1 | 0 | 1 |
| 500 | decode_smoke | disagg | tpot (ms/token) | 2 | 43.590500 | 44.163000 | -0.572500 | 1.281% | 2 | 1 | 0 | 1 |
| 500 | decode_smoke | disagg | ttft (ms) | 2 | 188.039000 | 280.622000 | -92.583000 | -33.802% | 2 | 2 | 0 | 0 |
| 500 | primary | agg | ranking_metric_value (tokens/s/gpu_cluster) | 4 | 2.294101e+06 | 2.706633e+06 | -412531.954551 | -22.266% | 4 | 2 | 0 | 2 |
| 500 | primary | agg | request_latency (ms) | 4 | 464.136234 | 445.763158 | 18.373075 | 5.408% | 4 | 2 | 0 | 2 |
| 500 | primary | agg | tpot (ms/token) | 4 | 123.940575 | 124.041364 | -0.100789 | 0.684% | 3 | 1 | 1 | 2 |
| 500 | primary | agg | ttft (ms) | 4 | 464.136234 | 445.763158 | 18.373075 | 5.408% | 4 | 2 | 0 | 2 |
| 500 | primary | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 4 | 19248.948504 | 14657.906997 | 4591.041507 | 41.042% | 4 | 3 | 0 | 1 |
| 500 | primary | disagg | request_latency (ms) | 4 | 382.571500 | 347.114250 | 35.457250 | 15.182% | 4 | 2 | 0 | 2 |
| 500 | primary | disagg | tpot (ms/token) | 4 | 0 | 0 | 0 | N/A | 0 | 0 | 4 | 0 |
| 500 | primary | disagg | ttft (ms) | 4 | 382.571500 | 347.114250 | 35.457250 | 15.182% | 4 | 2 | 0 | 2 |
| 5000 | decode_smoke | agg | ranking_metric_value (tokens/s/gpu_cluster) | 4 | 2276.645235 | 1854.001475 | 422.643760 | 61.295% | 4 | 4 | 0 | 0 |
| 5000 | decode_smoke | agg | request_latency (ms) | 4 | 178805.749669 | 217090.940319 | -38285.190650 | -21.772% | 4 | 4 | 0 | 0 |
| 5000 | decode_smoke | agg | tpot (ms/token) | 4 | 172.634931 | 210.910200 | -38.275269 | -22.165% | 4 | 4 | 0 | 0 |
| 5000 | decode_smoke | agg | ttft (ms) | 4 | 2200.215526 | 1329.805941 | 870.409585 | 40.541% | 4 | 1 | 0 | 3 |
| 5000 | decode_smoke | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 4 | 2125.112480 | 1656.849600 | 468.262880 | 146.333% | 4 | 4 | 0 | 0 |
| 5000 | decode_smoke | disagg | request_latency (ms) | 4 | 55261.705000 | 44208.807250 | 11052.897750 | 27.061% | 4 | 1 | 0 | 3 |
| 5000 | decode_smoke | disagg | tpot (ms/token) | 4 | 53.718500 | 42.160000 | 11.558500 | 30.137% | 4 | 1 | 0 | 3 |
| 5000 | decode_smoke | disagg | ttft (ms) | 4 | 307.679500 | 1079.127250 | -771.447750 | -55.522% | 4 | 4 | 0 | 0 |
| 5000 | primary | agg | ranking_metric_value (tokens/s/gpu_cluster) | 9 | 1.342415e+06 | 3.360415e+06 | -2.018001e+06 | 192.818% | 9 | 3 | 0 | 6 |
| 5000 | primary | agg | request_latency (ms) | 9 | 2700.924209 | 2594.570840 | 106.353369 | 21.216% | 9 | 4 | 0 | 5 |
| 5000 | primary | agg | tpot (ms/token) | 9 | 1416.419374 | 1535.919566 | -119.500191 | 20.047% | 9 | 5 | 0 | 4 |
| 5000 | primary | agg | ttft (ms) | 9 | 2700.924209 | 2594.570840 | 106.353369 | 21.216% | 9 | 4 | 0 | 5 |
| 5000 | primary | disagg | ranking_metric_value (tokens/s/gpu_cluster) | 9 | 12550.003867 | 8018.332337 | 4531.671530 | 534.104% | 9 | 8 | 0 | 1 |
| 5000 | primary | disagg | request_latency (ms) | 9 | 2784.361667 | 1575.761444 | 1208.600222 | 500.855% | 9 | 4 | 0 | 5 |
| 5000 | primary | disagg | tpot (ms/token) | 9 | 0 | 0 | 0 | N/A | 0 | 0 | 9 | 0 |
| 5000 | primary | disagg | ttft (ms) | 9 | 2784.361667 | 1575.761444 | 1208.600222 | 500.855% | 9 | 4 | 0 | 5 |

## TPOT Publication Status

| Status | Count | Meaning |
|---|---:|---|
| `computed` | 59 | Both values have a nonzero DeepSeek baseline; relative delta is numeric. |
| `zero_baseline_both_zero` | 30 | Step4 and DeepSeek are both zero; absolute delta is `0`, relative delta is `null`. |

TPOT is observational only and never changes TTFT eligibility or ranking eligibility.
