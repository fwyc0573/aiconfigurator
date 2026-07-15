# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SystemSpec — hardware system spec loaded from a per-system YAML file.

Subclasses ``dict`` so existing code that does ``spec["gpu"]["mem_bw"]`` or
``isinstance(spec, dict)`` keeps working. ``get_p2p_bandwidth`` is the only
added method, replacing ``PerfDatabase._get_p2p_bandwidth``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BandwidthSelection:
    """Immutable record of one topology bandwidth selection."""

    group_size: int
    tier: str
    bandwidth_bytes_per_second: int | float


class SystemSpec(dict):
    """Hardware system spec backed by the YAML dict.

    The dict is the single source of truth — there are no parallel structured
    attributes. Construct directly with ``SystemSpec(yaml_dict)``.
    """

    def select_p2p_bandwidth(self, num_gpus: int) -> BandwidthSelection:
        """Return the selected point-to-point bandwidth and topology tier.

        Three-tier selection:

        - ``num_gpus <= num_gpus_per_node``: ``intra_node_bw`` (NVLink within node)
        - ``num_gpus <= num_gpus_per_rack``: ``inter_node_bw`` (NVSwitch within rack)
        - ``num_gpus > num_gpus_per_rack``: ``inter_rack_bw`` (InfiniBand between racks),
          falling back to ``inter_node_bw`` when ``inter_rack_bw`` is unset.

        Raises ``KeyError`` for misconfigured specs that lack required keys —
        same loud-failure behavior as the original ``_get_p2p_bandwidth``.
        """
        node_spec = self["node"]
        num_gpus_per_node = node_spec["num_gpus_per_node"]
        num_gpus_per_rack = node_spec.get("num_gpus_per_rack", float("inf"))

        if num_gpus <= num_gpus_per_node:
            tier = "intra_node_bw"
            bandwidth = node_spec[tier]
        elif num_gpus <= num_gpus_per_rack:
            tier = "inter_node_bw"
            bandwidth = node_spec[tier]
        else:
            tier = "inter_rack_bw" if "inter_rack_bw" in node_spec else "inter_node_bw"
            bandwidth = node_spec[tier]

        return BandwidthSelection(
            group_size=num_gpus,
            tier=tier,
            bandwidth_bytes_per_second=bandwidth,
        )

    def get_p2p_bandwidth(self, num_gpus: int) -> float:
        """Return point-to-point bandwidth while preserving the legacy API."""
        return self.select_p2p_bandwidth(num_gpus).bandwidth_bytes_per_second
