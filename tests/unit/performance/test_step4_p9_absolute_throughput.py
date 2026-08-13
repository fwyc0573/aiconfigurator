"""Unit tests for the Step4 P9 absolute-throughput figure."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

pytestmark = pytest.mark.unit


def test_cli_plots_only_successes_and_preserves_all_terminal_gaps(tmp_path: Path):
    models = ("stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4")
    isls = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
    outcomes = []
    for model in models:
        for index, isl in enumerate(isls):
            if index < 4:
                outcomes.append(
                    {
                        "model": model,
                        "isl": isl,
                        "status": "success",
                        "throughput_per_used_gpu": 160.0 / (index + 1),
                    }
                )
            else:
                outcomes.append(
                    {
                        "model": model,
                        "isl": isl,
                        "status": "memory_infeasible" if index >= 6 else "sla_infeasible",
                        "reason": "terminal gap",
                    }
                )
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "schema": "step4-profiled-agg-matrix-v1",
                "status": "completed",
                "models": list(models),
                "isls": list(isls),
                "point_count": 16,
                "status_counts": {"success": 8, "memory_infeasible": 4, "sla_infeasible": 4},
                "terminal_outcomes": outcomes,
            }
        ),
        encoding="utf-8",
    )
    png_path = tmp_path / "absolute.png"
    manifest_path = tmp_path / "figure.json"
    env = os.environ.copy()
    env.update({"PYTHONPATH": "src:.", "MPLBACKEND": "Agg"})

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.performance.step4_p9_absolute_throughput",
            "--matrix",
            str(matrix_path),
            "--output",
            str(png_path),
            "--manifest",
            str(manifest_path),
        ],
        cwd=Path(__file__).parents[3],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "validated"
    assert manifest["plotted_success_count"] == 8
    assert manifest["terminal_gap_count"] == 8
    assert manifest["imputed_point_count"] == 0
    assert manifest["metric"] == "absolute output throughput per used GPU (token/s/GPU)"
    assert manifest["figure"]["dimensions_px"] == [2160, 1260]
    assert len(manifest["series"]) == 2
    assert all(len(series["points"]) == 8 for series in manifest["series"])
    assert (
        sum(point["throughput_per_used_gpu"] is None for series in manifest["series"] for point in series["points"])
        == 8
    )
    with Image.open(png_path) as image:
        assert image.size == (2160, 1260)
        assert image.format == "PNG"
