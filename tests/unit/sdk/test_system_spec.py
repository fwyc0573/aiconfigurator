# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for topology-aware SystemSpec bandwidth selection."""

from dataclasses import FrozenInstanceError

import pytest

from aiconfigurator.sdk import system_spec as system_spec_module


def _gb300_equal_bandwidth_spec():
    return system_spec_module.SystemSpec(
        {
            "node": {
                "num_gpus_per_node": 4,
                "num_gpus_per_rack": 72,
                "intra_node_bw": 900_000_000_000,
                "inter_node_bw": 900_000_000_000,
                "inter_rack_bw": 50_000_000_000,
            }
        }
    )


def _bandwidth_selection_type():
    selection_type = getattr(system_spec_module, "BandwidthSelection", None)
    assert selection_type is not None, "SystemSpec must expose BandwidthSelection"
    return selection_type


def test_select_p2p_bandwidth_preserves_group_and_tier_when_values_match():
    selection_type = _bandwidth_selection_type()
    spec = _gb300_equal_bandwidth_spec()

    group_2 = spec.select_p2p_bandwidth(2)
    group_8 = spec.select_p2p_bandwidth(8)

    assert group_2 == selection_type(
        group_size=2,
        tier="intra_node_bw",
        bandwidth_bytes_per_second=900_000_000_000,
    )
    assert group_8 == selection_type(
        group_size=8,
        tier="inter_node_bw",
        bandwidth_bytes_per_second=900_000_000_000,
    )
    assert group_2.bandwidth_bytes_per_second == group_8.bandwidth_bytes_per_second
    assert group_2.group_size != group_8.group_size
    assert group_2.tier != group_8.tier


def test_bandwidth_selection_is_immutable():
    selection_type = _bandwidth_selection_type()
    selection = selection_type(
        group_size=8,
        tier="inter_node_bw",
        bandwidth_bytes_per_second=900_000_000_000,
    )

    with pytest.raises(FrozenInstanceError):
        selection.group_size = 2


def test_get_p2p_bandwidth_remains_numeric_for_existing_callers():
    spec = _gb300_equal_bandwidth_spec()

    assert spec.get_p2p_bandwidth(2) == 900_000_000_000
    assert spec.get_p2p_bandwidth(8) == 900_000_000_000


def test_select_p2p_bandwidth_reports_the_actual_legacy_fallback_tier():
    spec = system_spec_module.SystemSpec(
        {
            "node": {
                "num_gpus_per_node": 4,
                "num_gpus_per_rack": 8,
                "intra_node_bw": 900_000_000_000,
                "inter_node_bw": 400_000_000_000,
            }
        }
    )

    selection = spec.select_p2p_bandwidth(16)

    assert selection.tier == "inter_node_bw"
    assert selection.bandwidth_bytes_per_second == 400_000_000_000
    assert spec.get_p2p_bandwidth(16) == 400_000_000_000
