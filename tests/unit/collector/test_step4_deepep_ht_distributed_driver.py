"""Contracts for the Step4-Pro DeepEP HT distributed collection driver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.performance.step4_pro_latest import (
    run_step4_deepep_ht_distributed as driver,
)

pytestmark = pytest.mark.unit


def test_case_selection_is_topology_exact_and_deterministic() -> None:
    ep16_full = driver.select_cases(ep_size=16, mode="full")
    ep32_full = driver.select_cases(ep_size=32, mode="full")
    ep16_smoke = driver.select_cases(ep_size=16, mode="smoke")

    assert len(ep16_full) == 29
    assert len(ep32_full) == 29
    assert len({tuple(case) for case in ep16_full + ep32_full}) == 58
    assert {int(case[1]) for case in ep16_full} == {16}
    assert {int(case[1]) for case in ep32_full} == {32}
    assert [int(case[6]) for case in ep16_smoke] == [1, 8192, 65536]


@pytest.mark.parametrize(
    ("ep_size", "mode", "match"),
    [
        (8, "smoke", "EP size"),
        (64, "full", "EP size"),
        (16, "sample", "mode"),
    ],
)
def test_case_selection_rejects_unrequested_population(
    ep_size: int,
    mode: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        driver.select_cases(ep_size=ep_size, mode=mode)


def test_validate_launcher_environment_requires_exact_ep_topology() -> None:
    valid = {
        "WORLD_SIZE": "16",
        "RANK": "11",
        "LOCAL_RANK": "3",
        "LOCAL_WORLD_SIZE": "8",
        "NODE_COUNT": "2",
        "NODE_RANK": "1",
        "PROC_PER_NODE": "8",
    }

    identity = driver.validate_launcher_environment(ep_size=16, environ=valid)

    assert identity == driver.LauncherIdentity(
        world_size=16,
        rank=11,
        local_rank=3,
        local_world_size=8,
        node_count=2,
        node_rank=1,
    )

    for key, value in {
        "WORLD_SIZE": "32",
        "LOCAL_WORLD_SIZE": "4",
        "NODE_COUNT": "4",
        "NODE_RANK": "0",
        "PROC_PER_NODE": "4",
    }.items():
        invalid = valid | {key: value}
        with pytest.raises(RuntimeError, match="launcher topology"):
            driver.validate_launcher_environment(ep_size=16, environ=invalid)


def test_execute_cases_emits_two_unique_rows_per_case(tmp_path: Path) -> None:
    cases = driver.select_cases(ep_size=16, mode="smoke")
    calls: list[tuple] = []

    def run_case(*case, perf_filename: str, device: str):
        calls.append((*case, perf_filename, device))
        tokens = int(case[6])
        return [
            {
                "operation": operation,
                "ep_size": 16,
                "tokens_per_dp_rank": tokens,
                "latency": tokens / 1000 + offset,
            }
            for operation, offset in (("dispatch", 0.1), ("combine", 0.2))
        ]

    rows = driver.execute_cases(
        cases=cases,
        run_case=run_case,
        perf_filename=str(tmp_path / "step4_deepep_ht_perf.txt"),
        local_rank=3,
    )

    assert len(calls) == 3
    assert all(call[-1] == "cuda:3" for call in calls)
    assert len(rows) == 6
    assert {(row["operation"], row["tokens_per_dp_rank"]) for row in rows} == {
        ("dispatch", 1),
        ("combine", 1),
        ("dispatch", 8192),
        ("combine", 8192),
        ("dispatch", 65536),
        ("combine", 65536),
    }


def test_result_summary_records_numeric_coverage(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    cases = driver.select_cases(ep_size=16, mode="smoke")
    rows = [
        {
            "operation": operation,
            "ep_size": 16,
            "tokens_per_dp_rank": int(case[6]),
            "latency": latency,
        }
        for case in cases
        for operation, latency in (("dispatch", 0.3), ("combine", 0.7))
    ]

    summary = driver.write_result_summary(
        summary_path=summary_path,
        ep_size=16,
        mode="smoke",
        cases=cases,
        rows=rows,
    )

    assert summary["completed_cases"] == 3
    assert summary["row_count"] == 6
    assert summary["operation_counts"] == {"combine": 3, "dispatch": 3}
    assert summary["token_min"] == 1
    assert summary["token_max"] == 65536
    assert summary["latency_min_ms"] == pytest.approx(0.3)
    assert summary["latency_max_ms"] == pytest.approx(0.7)
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary


def test_result_summary_rejects_missing_or_duplicate_physical_rows(
    tmp_path: Path,
) -> None:
    cases = driver.select_cases(ep_size=16, mode="smoke")
    rows = [
        {
            "operation": operation,
            "ep_size": 16,
            "tokens_per_dp_rank": int(case[6]),
            "latency": 0.5,
        }
        for case in cases
        for operation in ("dispatch", "combine")
    ]

    for invalid_rows in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(RuntimeError, match="physical rows"):
            driver.write_result_summary(
                summary_path=tmp_path / "invalid.json",
                ep_size=16,
                mode="smoke",
                cases=cases,
                rows=invalid_rows,
            )
