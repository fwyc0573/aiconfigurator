# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WideEP registry for pinned vLLM Step4-Pro-Latest."""

from collector.registry_types import OpEntry, PerfFile

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="step4_deepep_ht",
        module="collector.wideep.vllm.collect_step4_deepep_ht",
        get_func="get_step4_deepep_ht_test_cases",
        run_func="run_step4_deepep_ht",
        perf_filename=PerfFile.STEP4_DEEPEP_HT,
    ),
]
