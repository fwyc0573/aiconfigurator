"""Unit contracts for the Step4-Pro DeepEP HT NCCL preflight."""

from __future__ import annotations

import pytest

from tests.performance.step4_pro_latest import (
    run_step4_deepep_ht_nccl_preflight as preflight,
)

pytestmark = pytest.mark.unit


def test_validate_environment_requires_platform_hca_and_exact_topology() -> None:
    valid = {
        "NCCL_IB_HCA": "=mlx5_0,mlx5_1",
        "WORLD_SIZE": "16",
        "RANK": "11",
        "LOCAL_RANK": "3",
        "LOCAL_WORLD_SIZE": "8",
    }

    identity = preflight.validate_environment(valid)

    assert identity == preflight.PreflightIdentity(
        world_size=16,
        rank=11,
        local_rank=3,
        local_world_size=8,
        nccl_ib_hca="=mlx5_0,mlx5_1",
    )

    for key, value in {
        "NCCL_IB_HCA": "",
        "WORLD_SIZE": "8",
        "RANK": "16",
        "LOCAL_RANK": "8",
        "LOCAL_WORLD_SIZE": "4",
    }.items():
        invalid = valid | {key: value}
        with pytest.raises(RuntimeError):
            preflight.validate_environment(invalid)


@pytest.mark.parametrize(("world_size", "expected"), [(16, 136), (32, 528)])
def test_expected_rank_sum_is_exact(world_size: int, expected: int) -> None:
    assert preflight.expected_rank_sum(world_size) == expected
