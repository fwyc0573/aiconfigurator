"""Compute and plot Step4/DeepSeek throughput ratios from ranked results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from tests.performance.aic_roofline_pareto.run_dsv4pro_vs_step4_throughput import ISL_VALUES

MODEL_STEP4 = "stepfun-ai/Step4"
MODEL_DEEPSEEK = "deepseek-ai/DeepSeek-V4-Pro"
SYSTEM_DISPLAY = {
    "gb300": "gb300",
    "h200_sxm": "h200",
    "h100_sxm": "h100",
    "h800_sxm": "h800",
}


def _finite_positive(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field} must be numeric; got {value!r}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be finite; got {value!r}")
    if number <= 0:
        raise ValueError(f"{field} must be positive; got {value!r}")
    return number


def _rank_one_rows(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rank_one = [row for row in rows if row.get("rank") == 1]
    if not rank_one:
        raise ValueError("results contain no rank-one rows")
    return rank_one


def compute_throughput_ratios(
    rows: list[Mapping[str, Any]],
    *,
    require_complete: bool = False,
    allow_unpaired: bool = False,
) -> dict[str, dict[str, dict[int, float]]]:
    """Return ``step4 / DeepSeek-V4-Pro`` ratios for each GPU and ISL."""
    if require_complete and allow_unpaired:
        raise ValueError("require_complete and allow_unpaired cannot both be enabled")
    grouped: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in _rank_one_rows(rows):
        model = row.get("model")
        system = row.get("system")
        isl = row.get("isl")
        if model not in (MODEL_STEP4, MODEL_DEEPSEEK):
            raise ValueError(f"unexpected model in ranked rows: {model!r}")
        if system not in SYSTEM_DISPLAY:
            raise ValueError(f"unexpected system in ranked rows: {system!r}")
        if isl not in ISL_VALUES:
            raise ValueError(f"unexpected ISL in ranked rows: {isl!r}")
        key = (SYSTEM_DISPLAY[system], system, int(isl))
        if model in grouped[key]:
            raise ValueError(f"duplicate rank-one row for model={model!r}, system={system!r}, isl={isl}")
        grouped[key][model] = row

    ratios: dict[str, dict[str, dict[int, float]]] = {
        "prefill": defaultdict(dict),
        "decode": defaultdict(dict),
    }
    for (display_system, _system, isl), model_rows in grouped.items():
        if MODEL_STEP4 not in model_rows or MODEL_DEEPSEEK not in model_rows:
            if allow_unpaired:
                continue
            raise ValueError(f"missing model row for system={display_system!r}, isl={isl}")
        step4 = model_rows[MODEL_STEP4]
        deepseek = model_rows[MODEL_DEEPSEEK]
        for figure, field in (
            ("prefill", "prefill_input_throughput"),
            ("decode", "output_token_throughput"),
        ):
            numerator = _finite_positive(step4.get(field), field=f"Step4 {field}")
            denominator_value = deepseek.get(field)
            if denominator_value == 0:
                raise ValueError(f"zero DeepSeek-V4-Pro baseline for {field}")
            denominator = _finite_positive(denominator_value, field=f"DeepSeek-V4-Pro {field}")
            ratios[figure][display_system][isl] = numerator / denominator
    if require_complete:
        for figure in ratios:
            for system in SYSTEM_DISPLAY.values():
                if set(ratios[figure][system]) != set(ISL_VALUES):
                    raise ValueError(f"incomplete {figure} ratios for system={system!r}")
    return ratios


def _display_systems_in_rows(rows: list[Mapping[str, Any]]) -> tuple[str, ...]:
    systems: list[str] = []
    for row in _rank_one_rows(rows):
        system = row.get("system")
        if system not in SYSTEM_DISPLAY:
            raise ValueError(f"unexpected system in ranked rows: {system!r}")
        display_system = SYSTEM_DISPLAY[system]
        if display_system not in systems:
            systems.append(display_system)
    if not systems:
        raise ValueError("results contain no recognized systems")
    return tuple(systems)


def _missing_ratio_points(rows: list[Mapping[str, Any]], systems: Iterable[str]) -> list[dict[str, Any]]:
    present_models: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in _rank_one_rows(rows):
        present_models[(str(row["system"]), int(row["isl"]))].add(str(row["model"]))

    missing_points = []
    display_to_system = {display: system for system, display in SYSTEM_DISPLAY.items()}
    for display_system in systems:
        system = display_to_system[display_system]
        for isl in ISL_VALUES:
            missing_models = [
                model for model in (MODEL_STEP4, MODEL_DEEPSEEK) if model not in present_models[(system, isl)]
            ]
            if missing_models:
                missing_points.append(
                    {
                        "system": display_system,
                        "isl": isl,
                        "missing_models": missing_models,
                    }
                )
    return missing_points


def _spread_annotation_positions(positions: list[float], *, minimum_gap: float) -> list[float]:
    """Spread ordered pixel positions while preserving their center."""
    if minimum_gap <= 0:
        raise ValueError(f"minimum_gap must be positive; got {minimum_gap!r}")
    if len(positions) < 2:
        return list(positions)

    adjusted = list(positions)
    for index in range(1, len(adjusted)):
        adjusted[index] = max(adjusted[index], adjusted[index - 1] + minimum_gap)
    center_shift = (sum(adjusted) - sum(positions)) / len(adjusted)
    return [position - center_shift for position in adjusted]


def _x_axis_limits() -> tuple[float, float]:
    return ISL_VALUES[0] / 1.25, ISL_VALUES[-1] * 1.25


def _plot_one(
    ratios: Mapping[str, Mapping[int, float]],
    *,
    systems: Iterable[str],
    title: str,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(12, 7))
    colors = {"gb300": "tab:blue", "h200": "tab:orange", "h100": "tab:green", "h800": "tab:red"}
    for system in systems:
        values = [ratios[system].get(isl, float("nan")) for isl in ISL_VALUES]
        axis.plot(ISL_VALUES, values, marker="o", linewidth=2, label=system, color=colors[system])
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Step4 baseline")
    axis.set_xscale("log", base=2)
    axis.set_xlim(*_x_axis_limits())
    axis.margins(y=0.12)
    axis.set_xticks(ISL_VALUES)
    axis.set_xticklabels([str(value) for value in ISL_VALUES])
    axis.set_xlabel("Sequence length (tokens)")
    axis.set_ylabel("Relative throughput vs step4 (step4 / DeepSeek-V4-Pro)")
    axis.set_title(title)
    axis.grid(True, which="both", linestyle="--", linewidth=0.6, color="lightgray")
    axis.legend(loc="upper left")
    fig.tight_layout()
    fig.canvas.draw()

    for isl in ISL_VALUES:
        points = sorted(
            ((system, ratios[system][isl]) for system in systems if isl in ratios[system]),
            key=lambda item: item[1],
        )
        pixel_positions = [axis.transData.transform((isl, value))[1] for _, value in points]
        label_positions = _spread_annotation_positions(pixel_positions, minimum_gap=18.0)
        for (system, value), point_y, label_y in zip(
            points,
            pixel_positions,
            label_positions,
            strict=True,
        ):
            axis.annotate(
                f"{system} {value:.3f}",
                (isl, value),
                textcoords="offset points",
                xytext=(0, (label_y - point_y) * 72.0 / fig.dpi),
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _load_rank_one_rows(results_paths: Iterable[str | Path]) -> tuple[tuple[Path, ...], list[Mapping[str, Any]]]:
    sources = tuple(Path(path) for path in results_paths)
    if not sources:
        raise ValueError("at least one results.json path is required")
    rank_one_rows: list[Mapping[str, Any]] = []
    for source in sources:
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = payload.get("ranked_rows")
        if not isinstance(rows, list):
            raise TypeError(f"{source} must contain a ranked_rows list")
        rank_one_rows.extend(_rank_one_rows(rows))
    return sources, rank_one_rows


def generate_figures(
    results_paths: str | Path | Iterable[str | Path],
    output_dir: str | Path,
    *,
    combined_output: str | Path | None = None,
) -> dict[str, Path]:
    """Read one or more runner artifacts and write combined results plus figures."""
    paths = (results_paths,) if isinstance(results_paths, str | Path) else results_paths
    sources, rows = _load_rank_one_rows(paths)
    systems = _display_systems_in_rows(rows)
    ratios = compute_throughput_ratios(rows, allow_unpaired=True)
    missing_ratio_points = _missing_ratio_points(rows, systems)
    paired_ratio_point_count = sum(len(system_ratios) for system_ratios in ratios["prefill"].values())
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "prefill": directory / "fig1_prefill_throughput_ratio.png",
        "decode": directory / "fig2_decode_throughput_ratio.png",
    }
    _plot_one(
        ratios["prefill"],
        systems=systems,
        title="Prefill throughput ratio vs baseline (step4)",
        output_path=figure_paths["prefill"],
    )
    _plot_one(
        ratios["decode"],
        systems=systems,
        title="Decode throughput ratio vs baseline (step4)",
        output_path=figure_paths["decode"],
    )
    if combined_output is not None:
        combined_path = Path(combined_output)
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined_payload = {
            "summary": {
                "source_count": len(sources),
                "rank_one_row_count": len(rows),
                "paired_ratio_point_count": paired_ratio_point_count,
                "missing_ratio_point_count": len(missing_ratio_points),
                "systems": list(systems),
                "ratio_direction": "step4 / DeepSeek-V4-Pro",
            },
            "source_results": [str(source) for source in sources],
            "ranked_rows": rows,
            "missing_ratio_points": missing_ratio_points,
            "ratios": ratios,
        }
        combined_path.write_text(json.dumps(combined_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return figure_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--combined-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = generate_figures(args.results_json, args.output_dir, combined_output=args.combined_output)
    print(f"wrote prefill={paths['prefill']} decode={paths['decode']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
