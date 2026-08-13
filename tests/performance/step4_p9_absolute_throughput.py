"""Plot Step4 V3/V4 absolute per-used-GPU throughput with terminal gaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

EXPECTED_MODELS = ("stepfun-ai/Step4-Pro-V3", "stepfun-ai/Step4-Pro-V4")
EXPECTED_ISLS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
TERMINAL_STATUSES = {"success", "memory_infeasible", "sla_infeasible"}
MODEL_LABELS = {
    "stepfun-ai/Step4-Pro-V3": "Step4-Pro-V3",
    "stepfun-ai/Step4-Pro-V4": "Step4-Pro-V4",
}
MODEL_COLORS = {
    "stepfun-ai/Step4-Pro-V3": "#2563EB",
    "stepfun-ai/Step4-Pro-V4": "#DC2626",
}
STATUS_STYLES = {
    "success": ("#16A34A", "o", "Success"),
    "memory_infeasible": ("#6B7280", "X", "Memory infeasible"),
    "sla_infeasible": ("#D97706", "D", "SLA infeasible"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_matrix(path: Path) -> dict[str, Any]:
    matrix = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(matrix, dict):
        raise TypeError("matrix artifact must contain a JSON object")
    if matrix.get("status") != "completed":
        raise ValueError("matrix must be completed without hard errors")
    if tuple(matrix.get("models", [])) != EXPECTED_MODELS:
        raise ValueError("matrix model order does not match the fixed Step4 contract")
    if tuple(matrix.get("isls", [])) != EXPECTED_ISLS:
        raise ValueError("matrix ISL order does not match the fixed Step4 contract")
    if matrix.get("point_count") != len(EXPECTED_MODELS) * len(EXPECTED_ISLS):
        raise ValueError("matrix point_count must be 16")
    return matrix


def _normalize_points(matrix: dict[str, Any]) -> tuple[list[dict[str, Any]], Counter[str]]:
    outcomes = matrix.get("terminal_outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 16:
        raise ValueError("matrix must contain exactly 16 terminal outcomes")
    by_identity: dict[tuple[str, int], dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    points = []
    for outcome in outcomes:
        model = outcome.get("model")
        isl = outcome.get("isl")
        status = outcome.get("status")
        identity = (model, isl)
        if model not in EXPECTED_MODELS or isl not in EXPECTED_ISLS:
            raise ValueError(f"outcome is outside the fixed matrix: {identity}")
        if identity in by_identity:
            raise ValueError(f"duplicate matrix outcome: {identity}")
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"unsupported P9 terminal status: {status}")
        throughput = outcome.get("throughput_per_used_gpu")
        if status == "success":
            if not isinstance(throughput, (int, float)) or not math.isfinite(throughput) or throughput <= 0:
                raise ValueError(f"success throughput must be finite and positive: {identity}")
            normalized_throughput = float(throughput)
        else:
            if throughput is not None:
                raise ValueError(f"terminal gap must not carry fabricated throughput: {identity}")
            normalized_throughput = None
        point = {
            "model": model,
            "isl": isl,
            "status": status,
            "throughput_per_used_gpu": normalized_throughput,
        }
        if status != "success":
            point["reason"] = str(outcome.get("reason", ""))
        by_identity[identity] = point
        counts[status] += 1

    for model in EXPECTED_MODELS:
        for isl in EXPECTED_ISLS:
            points.append(by_identity[(model, isl)])
    declared_counts = Counter({key: int(value) for key, value in matrix.get("status_counts", {}).items()})
    if counts != declared_counts:
        raise ValueError(f"matrix status counts disagree with terminal outcomes: {counts} != {declared_counts}")
    return points, counts


def _plot(points: list[dict[str, Any]], output_path: Path) -> None:
    x = np.arange(len(EXPECTED_ISLS))
    fig = plt.figure(figsize=(12, 7), dpi=180, constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(5, 1.25))
    throughput_ax = fig.add_subplot(grid[0])
    status_ax = fig.add_subplot(grid[1], sharex=throughput_ax)

    for model in EXPECTED_MODELS:
        model_points = [point for point in points if point["model"] == model]
        throughput = [
            point["throughput_per_used_gpu"] if point["throughput_per_used_gpu"] is not None else np.nan
            for point in model_points
        ]
        throughput_ax.plot(
            x,
            throughput,
            color=MODEL_COLORS[model],
            linewidth=2.25,
            marker="o",
            markersize=6,
            label=MODEL_LABELS[model],
        )
        for index, point in enumerate(model_points):
            value = point["throughput_per_used_gpu"]
            if value is not None:
                throughput_ax.annotate(
                    f"{value:.1f}",
                    (index, value),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.5,
                    color=MODEL_COLORS[model],
                )

    throughput_ax.set_yscale("log")
    throughput_ax.set_ylabel("Absolute output throughput (token/s/used-GPU)")
    throughput_ax.set_title("Step4-Pro H800 Aggregate Serving — Measured-Kernel-Backed Mixed AIC Prediction")
    throughput_ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.32)
    throughput_ax.legend(loc="upper right", frameon=False)
    throughput_ax.tick_params(axis="x", labelbottom=False)

    for row, model in enumerate(EXPECTED_MODELS):
        model_points = [point for point in points if point["model"] == model]
        for index, point in enumerate(model_points):
            color, marker, _ = STATUS_STYLES[point["status"]]
            status_ax.scatter(index, row, color=color, marker=marker, s=54, zorder=3)
    status_ax.set_yticks(range(len(EXPECTED_MODELS)), [MODEL_LABELS[model] for model in EXPECTED_MODELS])
    status_ax.set_xticks(x, [f"{isl:,}" for isl in EXPECTED_ISLS], rotation=30, ha="right")
    status_ax.set_xlabel("Input sequence length (tokens); missing throughput points are explicit terminal gaps")
    status_ax.set_ylim(-0.65, len(EXPECTED_MODELS) - 0.35)
    status_ax.grid(True, axis="x", linestyle=":", alpha=0.3)
    status_ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=color,
                markeredgecolor=color,
                label=label,
                markersize=7,
            )
            for color, marker, label in STATUS_STYLES.values()
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.6),
        ncol=3,
        frameon=False,
    )
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def build_figure(*, matrix_path: Path, output_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Validate the P8 matrix, plot real successes, and preserve terminal gaps."""
    matrix = _load_matrix(matrix_path)
    points, counts = _normalize_points(matrix)
    _plot(points, output_path)
    series = [
        {
            "model": model,
            "label": MODEL_LABELS[model],
            "points": [point for point in points if point["model"] == model],
        }
        for model in EXPECTED_MODELS
    ]
    manifest = {
        "schema": "step4-p9-absolute-throughput-figure-v1",
        "status": "validated",
        "metric": "absolute output throughput per used GPU (token/s/GPU)",
        "provenance": "measured-kernel-backed mixed AIC prediction; not end-to-end hardware measurement",
        "models": list(EXPECTED_MODELS),
        "isls": list(EXPECTED_ISLS),
        "status_counts": dict(sorted(counts.items())),
        "plotted_success_count": counts["success"],
        "terminal_gap_count": len(points) - counts["success"],
        "imputed_point_count": 0,
        "series": series,
        "source_matrix": {"path": str(matrix_path), "sha256": _sha256(matrix_path)},
        "figure": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "dimensions_px": [2160, 1260],
            "format": "PNG",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    build_figure(matrix_path=args.matrix, output_path=args.output, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
