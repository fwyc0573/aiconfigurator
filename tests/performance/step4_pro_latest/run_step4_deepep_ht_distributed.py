#!/usr/bin/env python3
"""Run one exact Step4-Pro DeepEP HT topology under torchrun."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SMOKE_TOKENS = (1, 8192, 65536)
SUPPORTED_EP_SIZES = (16, 32)
EP_RANKS_PER_NODE = 8


@dataclass(frozen=True)
class LauncherIdentity:
    """Validated torchrun and platform topology."""

    world_size: int
    rank: int
    local_rank: int
    local_world_size: int
    node_count: int
    node_rank: int


def select_cases(*, ep_size: int, mode: str) -> list[list[object]]:
    """Select the exact topology population in deterministic token order."""
    if ep_size not in SUPPORTED_EP_SIZES:
        raise ValueError(f"Step4 DeepEP HT EP size must be one of {SUPPORTED_EP_SIZES}, got {ep_size}")
    if mode not in {"smoke", "full"}:
        raise ValueError(f"Step4 DeepEP HT mode must be 'smoke' or 'full', got {mode!r}")

    from collector.wideep.vllm.collect_step4_deepep_ht import (
        get_step4_deepep_ht_test_cases,
    )

    cases = [case for case in get_step4_deepep_ht_test_cases() if int(case[1]) == ep_size]
    if mode == "smoke":
        cases = [case for case in cases if int(case[6]) in SMOKE_TOKENS]

    expected_count = len(SMOKE_TOKENS) if mode == "smoke" else 29
    if len(cases) != expected_count:
        raise RuntimeError(
            "Step4 DeepEP HT selected an unexpected case population: "
            f"ep_size={ep_size}, mode={mode}, expected={expected_count}, "
            f"actual={len(cases)}"
        )
    if len(cases) != len({tuple(case) for case in cases}):
        raise RuntimeError("Step4 DeepEP HT selected duplicate distributed cases")
    return cases


def validate_launcher_environment(
    *,
    ep_size: int,
    environ: Mapping[str, str],
) -> LauncherIdentity:
    """Require torchrun and platform topology to describe the same EP job."""
    required = (
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "NODE_COUNT",
        "NODE_RANK",
        "PROC_PER_NODE",
    )
    missing = [name for name in required if name not in environ]
    if missing:
        raise RuntimeError(f"Step4 DeepEP HT launcher topology is incomplete; missing environment variables: {missing}")

    values = {name: int(environ[name]) for name in required}
    expected_nodes = ep_size // EP_RANKS_PER_NODE
    valid = (
        ep_size in SUPPORTED_EP_SIZES
        and values["WORLD_SIZE"] == ep_size
        and values["LOCAL_WORLD_SIZE"] == EP_RANKS_PER_NODE
        and values["PROC_PER_NODE"] == EP_RANKS_PER_NODE
        and values["NODE_COUNT"] == expected_nodes
        and 0 <= values["RANK"] < ep_size
        and 0 <= values["LOCAL_RANK"] < EP_RANKS_PER_NODE
        and 0 <= values["NODE_RANK"] < expected_nodes
        and values["RANK"] // EP_RANKS_PER_NODE == values["NODE_RANK"]
        and values["RANK"] % EP_RANKS_PER_NODE == values["LOCAL_RANK"]
    )
    if not valid:
        raise RuntimeError(
            f"Step4 DeepEP HT launcher topology does not match the requested EP job: ep_size={ep_size}, values={values}"
        )

    return LauncherIdentity(
        world_size=values["WORLD_SIZE"],
        rank=values["RANK"],
        local_rank=values["LOCAL_RANK"],
        local_world_size=values["LOCAL_WORLD_SIZE"],
        node_count=values["NODE_COUNT"],
        node_rank=values["NODE_RANK"],
    )


def execute_cases(
    *,
    cases: Sequence[Sequence[object]],
    run_case: Callable[..., list[dict[str, Any]]],
    perf_filename: str,
    local_rank: int,
) -> list[dict[str, Any]]:
    """Execute every selected case and validate its two returned physical rows."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for case in cases:
        case_rows = run_case(
            *case,
            perf_filename=perf_filename,
            device=f"cuda:{local_rank}",
        )
        expected_tokens = int(case[6])
        expected_ep_size = int(case[1])
        if len(case_rows) != 2:
            raise RuntimeError(
                "Step4 DeepEP HT case did not return two physical rows: "
                f"ep_size={expected_ep_size}, tokens={expected_tokens}, "
                f"rows={len(case_rows)}"
            )
        for row in case_rows:
            operation = str(row.get("operation"))
            key = (operation, int(row.get("tokens_per_dp_rank", -1)))
            latency = float(row.get("latency", float("nan")))
            if (
                operation not in {"dispatch", "combine"}
                or int(row.get("ep_size", -1)) != expected_ep_size
                or key[1] != expected_tokens
                or not math.isfinite(latency)
                or latency <= 0.0
                or key in seen
            ):
                raise RuntimeError(
                    f"Step4 DeepEP HT returned an invalid physical row: case={tuple(case)!r}, row={row!r}"
                )
            seen.add(key)
            rows.append(row)
    return rows


def write_result_summary(
    *,
    summary_path: Path,
    ep_size: int,
    mode: str,
    cases: Sequence[Sequence[object]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate complete dispatch/combine coverage and write numeric evidence."""
    expected = {(operation, int(case[6])) for case in cases for operation in ("dispatch", "combine")}
    actual = [(str(row.get("operation")), int(row.get("tokens_per_dp_rank", -1))) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError(
            "Step4 DeepEP HT physical rows are incomplete or duplicated: "
            f"expected={len(expected)}, actual={len(actual)}, "
            f"unique_actual={len(set(actual))}"
        )

    latencies = [float(row["latency"]) for row in rows]
    if any(not math.isfinite(value) or value <= 0.0 for value in latencies):
        raise RuntimeError("Step4 DeepEP HT physical rows contain invalid latency")
    tokens = [int(case[6]) for case in cases]
    summary = {
        "status": "PASS",
        "ep_size": ep_size,
        "ep_ranks_per_node": EP_RANKS_PER_NODE,
        "node_count": ep_size // EP_RANKS_PER_NODE,
        "mode": mode,
        "completed_cases": len(cases),
        "row_count": len(rows),
        "operation_counts": dict(sorted(Counter(key[0] for key in actual).items())),
        "token_min": min(tokens),
        "token_max": max(tokens),
        "latency_min_ms": min(latencies),
        "latency_max_ms": max(latencies),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = validate_launcher_environment(
        ep_size=args.ep_size,
        environ=os.environ,
    )
    cases = select_cases(ep_size=args.ep_size, mode=args.mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    perf_path = args.output_dir / "step4_deepep_ht_perf.txt"
    summary_path = args.output_dir / "step4_deepep_ht_summary.json"
    if identity.rank == 0 and (perf_path.exists() or summary_path.exists()):
        raise RuntimeError(f"Step4 DeepEP HT output path must be fresh before collection: {args.output_dir}")

    from collector.wideep.vllm.collect_step4_deepep_ht import (
        run_step4_deepep_ht,
    )

    rows = execute_cases(
        cases=cases,
        run_case=run_step4_deepep_ht,
        perf_filename=str(perf_path),
        local_rank=identity.local_rank,
    )

    import torch.distributed as dist

    dist.barrier()
    if identity.rank == 0:
        if not perf_path.is_file() or perf_path.stat().st_size == 0:
            raise RuntimeError(f"Step4 DeepEP HT rank 0 did not create the perf file: {perf_path}")
        summary = write_result_summary(
            summary_path=summary_path,
            ep_size=args.ep_size,
            mode=args.mode,
            cases=cases,
            rows=rows,
        )
        print("STEP4_DEEPEP_HT_DISTRIBUTED=PASS " + " ".join(f"{key}={value}" for key, value in summary.items()))
    dist.barrier()
    print("STEP4_DEEPEP_HT_RANK=PASS " + " ".join(f"{key}={value}" for key, value in asdict(identity).items()))


if __name__ == "__main__":
    main()
