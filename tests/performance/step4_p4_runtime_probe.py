"""P4 worker-side runtime identity and attention-fidelity probe.

This script is intended to run inside the exact H800 image.  It fails fast on
missing dependencies or mismatched runtime facts; it never substitutes a
different framework, device, or historical data source.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import statistics
import subprocess
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

REQUIRED_VLLM_VERSION = "0.19.0"
REQUIRED_SM = (9, 0)
REQUIRED_PHYSICAL_FLOOR_SOURCE = "derived_h800_peak_and_traffic"
_TREND_Z = 1.96
_FLASH_PROVIDERS = {"top_level", "vllm_bundled"}
_FLASH_BACKEND = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
_FLASH_MODULE_BY_PROVIDER = {
    "top_level": "flash_attn",
    "vllm_bundled": "vllm.vllm_flash_attn",
}
_BUNDLED_FLASH_VERSION = "3"
_WINDOW_SIZE = 512
_LONG_CONTEXT_MULTIPLIER = 2
_MIN_SAMPLE_COUNT = 3
STEP4_SWA_PROFILE = {
    "kernel_source": "vllm_flash_attn",
    "batch_size": 1,
    "num_heads": 128,
    "num_key_value_heads": 16,
    "head_dim": 128,
    "attn_dtype": "bfloat16",
    "kv_cache_dtype": "fp8",
}


def _derive_step4_swa_physical_inputs(row: dict) -> dict:
    """Derive the mandatory Step4 SWA work and traffic from row geometry."""

    phase = row["phase"]
    sequence = int(row["sequence"])
    window = int(row["window_size"])
    head_dim = STEP4_SWA_PROFILE["head_dim"]
    heads = STEP4_SWA_PROFILE["num_heads"]
    kv_heads = STEP4_SWA_PROFILE["num_key_value_heads"]
    batch = STEP4_SWA_PROFILE["batch_size"]
    if phase == "context":
        if window == 0:
            compute_flops = 2 * head_dim * batch * heads * sequence * (sequence + 1)
            effective_tokens = sequence
        else:
            compute_flops = 4 * head_dim * batch * heads * sum(min(window, index + 1) for index in range(sequence))
            effective_tokens = min(sequence, window)
        traffic_bytes = 4 * head_dim * heads * sequence + 4 * head_dim * kv_heads * effective_tokens
    else:
        effective_tokens = sequence if window == 0 else min(sequence, window)
        compute_flops = 4 * head_dim * batch * heads * effective_tokens
        traffic_bytes = 4 * head_dim * heads + 2 * head_dim * kv_heads * effective_tokens + 2 * head_dim * kv_heads
    return {
        "compute_flops": compute_flops,
        "tc_flops": 989_000_000_000_000,
        "traffic_bytes": traffic_bytes,
        "mem_bw_bytes_per_s": 3_350_000_000_000,
    }


def validate_image_provenance(reference: str, manifest_digest: str, config_digest: str) -> dict:
    """Validate the immutable image identity recorded by the P4 artifact."""

    if not reference.strip():
        raise ValueError("image reference must be non-empty")
    for label, digest in (("manifest", manifest_digest), ("config", config_digest)):
        prefix = "sha256:"
        value = str(digest)
        suffix = value[len(prefix) :] if value.startswith(prefix) else ""
        if len(suffix) != 64 or any(char not in "0123456789abcdef" for char in suffix):
            raise ValueError(f"image {label} digest must be a lowercase SHA-256")
    return {
        "reference": reference,
        "manifest_digest": manifest_digest,
        "config_digest": config_digest,
    }


def _flash_attention_identity() -> dict:
    """Resolve and fingerprint the actual FlashAttention implementation."""

    try:
        flash_module = importlib.import_module("flash_attn")
        provider = "top_level"
    except ModuleNotFoundError as exc:
        if exc.name != "flash_attn":
            raise
        flash_module = importlib.import_module("vllm.vllm_flash_attn")
        provider = "vllm_bundled"

    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    backend_class = AttentionBackendEnum.FLASH_ATTN.get_class()
    backend = f"{backend_class.__module__}.{backend_class.__name__}"
    if backend != _FLASH_BACKEND or backend_class.get_name() != "FLASH_ATTN":
        raise RuntimeError(f"unexpected FlashAttention backend resolution: {backend!r}")

    if provider == "top_level":
        flash_version = str(getattr(flash_module, "__version__", "")) or None
    else:
        backend_module = importlib.import_module(backend_class.__module__)
        get_flash_attn_version = getattr(backend_module, "get_flash_attn_version", None)
        if get_flash_attn_version is None:
            raise RuntimeError("bundled FlashAttention backend does not expose get_flash_attn_version")
        flash_version = str(get_flash_attn_version())

    module_file = Path(flash_module.__file__).resolve()
    if not module_file.is_file():
        raise RuntimeError(f"FlashAttention module has no file: {module_file}")
    digest = hashlib.sha256()
    package_root = module_file.parent
    for path in sorted(path for path in package_root.rglob("*") if path.is_file()):
        digest.update(str(path.relative_to(package_root)).encode("utf-8"))
        digest.update(path.read_bytes())

    return {
        "flash_attn_provider": provider,
        "flash_attn_module": str(flash_module.__name__),
        "flash_attn_backend": backend,
        "flash_attn_version": flash_version,
        "flash_attn_provenance_sha256": digest.hexdigest(),
    }


def validate_runtime_facts(facts: dict) -> None:
    """Validate the exact runtime identity required by the P4 gate."""

    if facts.get("vllm_version") != REQUIRED_VLLM_VERSION:
        raise ValueError(f"vLLM version must be {REQUIRED_VLLM_VERSION}, got {facts.get('vllm_version')!r}")
    device_name = str(facts.get("device_name", ""))
    if "H800" not in device_name.upper():
        raise ValueError(f"runtime device must be H800, got {device_name!r}")
    if tuple(facts.get("compute_capability", ())) != REQUIRED_SM:
        raise ValueError(f"runtime device must be SM90, got {facts.get('compute_capability')!r}")
    required_values = (
        "cuda_version",
        "nccl_version",
        "flash_attn_provider",
        "flash_attn_module",
        "flash_attn_backend",
        "flash_attn_version",
        "flash_attn_provenance_sha256",
        "sm_clock_mhz",
        "memory_clock_mhz",
    )
    missing = [key for key in required_values if facts.get(key) in (None, "")]
    if missing:
        raise ValueError(f"runtime facts missing required fields: {missing}")
    if facts["flash_attn_provider"] not in _FLASH_PROVIDERS:
        raise ValueError(f"unsupported FlashAttention provider: {facts['flash_attn_provider']!r}")
    expected_module = _FLASH_MODULE_BY_PROVIDER[facts["flash_attn_provider"]]
    if facts["flash_attn_module"] != expected_module:
        raise ValueError(
            f"FlashAttention provider {facts['flash_attn_provider']!r} requires module {expected_module!r}"
        )
    if facts["flash_attn_provider"] == "vllm_bundled" and str(facts["flash_attn_version"]) != _BUNDLED_FLASH_VERSION:
        raise ValueError(f"bundled FlashAttention version must be {_BUNDLED_FLASH_VERSION}")
    if facts["flash_attn_backend"] != _FLASH_BACKEND:
        raise ValueError(f"runtime FlashAttention backend must be {_FLASH_BACKEND!r}")
    provenance = str(facts["flash_attn_provenance_sha256"])
    if len(provenance) != 64 or any(char not in "0123456789abcdef" for char in provenance):
        raise ValueError("FlashAttention provenance must be a lowercase SHA-256")
    for key in ("sm_clock_mhz", "memory_clock_mhz"):
        if float(facts[key]) <= 0:
            raise ValueError(f"runtime clock {key} must be positive, got {facts[key]!r}")


def _query_clock_mhz() -> tuple[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    first_line = output.splitlines()[0]
    values = [int(value.strip()) for value in first_line.split(",")]
    if len(values) != 2:
        raise RuntimeError(f"unexpected nvidia-smi clock output: {output!r}")
    return values[0], values[1]


def collect_runtime_facts() -> dict:
    """Collect and validate runtime facts inside the target worker."""

    import torch
    import vllm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable on the profiling worker")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    nccl_version = torch.cuda.nccl.version()
    if isinstance(nccl_version, tuple):
        nccl_version = ".".join(str(value) for value in nccl_version)
    sm_clock_mhz, memory_clock_mhz = _query_clock_mhz()
    flash_identity = _flash_attention_identity()
    facts = {
        "vllm_version": str(vllm.__version__),
        "device_name": str(properties.name),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "cuda_version": str(torch.version.cuda),
        "nccl_version": str(nccl_version),
        "sm_clock_mhz": sm_clock_mhz,
        "memory_clock_mhz": memory_clock_mhz,
        "gpu_index": 0,
        "total_memory_bytes": int(properties.total_memory),
    }
    facts.update(flash_identity)
    validate_runtime_facts(facts)
    return facts


def _median_and_variance(samples: list[float]) -> tuple[float, float]:
    if not samples:
        raise ValueError("attention sample list cannot be empty")
    if len(samples) < _MIN_SAMPLE_COUNT:
        raise ValueError(f"attention fidelity requires at least {_MIN_SAMPLE_COUNT} samples per row")
    values = [float(value) for value in samples]
    invalid = [value for value in values if not math.isfinite(value) or value <= 0]
    if invalid:
        raise ValueError(f"attention samples must be finite and positive, got {invalid!r}")
    return statistics.median(values), statistics.variance(values) if len(values) > 1 else 0.0


def summarize_attention_fidelity(rows: list[dict], *, required_profile: dict | None = None) -> dict:
    """Validate matched context/generation window pairs and return metrics."""

    if not rows:
        raise ValueError("attention fidelity requires non-empty rows")
    physical_floor_violations = []
    grouped: dict[tuple, dict[int, dict]] = defaultdict(dict)
    phases = set()
    for row in rows:
        if required_profile is not None:
            mismatches = {
                field: {"expected": expected, "actual": row.get(field)}
                for field, expected in required_profile.items()
                if row.get(field) != expected
            }
            if mismatches:
                raise ValueError(f"Step4 SWA profile mismatch: {mismatches}")
        phase = row.get("phase")
        window = row.get("window_size")
        if phase not in {"context", "generation"} or window not in {0, 512}:
            raise ValueError(f"unsupported attention fidelity row axes: phase={phase!r}, window={window!r}")
        try:
            latency = float(row["latency_ms"])
            floor = float(row["physical_floor_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"attention latency/floor must be numeric: row={row!r}") from exc
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"attention latency must be finite and positive, got {latency!r}")
        sample_median, _ = _median_and_variance(row.get("samples_ms"))
        if not math.isclose(latency, sample_median, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"latency_ms must equal samples median: reported={latency:.9g}, median={sample_median:.9g}"
            )
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError(f"physical floor must be finite and positive, got {floor!r}")
        if row.get("physical_floor_source") != REQUIRED_PHYSICAL_FLOOR_SOURCE:
            raise ValueError(
                "physical_floor_source must be "
                f"{REQUIRED_PHYSICAL_FLOOR_SOURCE!r}, got {row.get('physical_floor_source')!r}"
            )
        inputs = row.get("physical_floor_inputs")
        if not isinstance(inputs, dict):
            raise TypeError("physical_floor_inputs must be a mapping")
        try:
            compute_flops = float(inputs["compute_flops"])
            tc_flops = float(inputs["tc_flops"])
            traffic_bytes = float(inputs["traffic_bytes"])
            mem_bw_bytes_per_s = float(inputs["mem_bw_bytes_per_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"physical_floor_inputs missing numeric fields: {inputs!r}") from exc
        floor_inputs = (compute_flops, tc_flops, traffic_bytes, mem_bw_bytes_per_s)
        if any(not math.isfinite(value) or value <= 0 for value in floor_inputs):
            raise ValueError(f"physical_floor_inputs must be finite and positive: {inputs!r}")
        if required_profile is not None:
            expected_inputs = _derive_step4_swa_physical_inputs(row)
            mismatched_inputs = {
                field: {"expected": expected, "actual": inputs.get(field)}
                for field, expected in expected_inputs.items()
                if float(inputs.get(field, math.nan)) != float(expected)
            }
            if mismatched_inputs:
                raise ValueError(
                    f"physical floor does not match derived Step4 SWA physical inputs: {mismatched_inputs}"
                )
        compute_floor_ms = compute_flops / tc_flops * 1000.0
        traffic_floor_ms = traffic_bytes / mem_bw_bytes_per_s * 1000.0
        expected_floor_ms = max(compute_floor_ms, traffic_floor_ms)
        if not math.isclose(floor, expected_floor_ms, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(
                "physical floor does not match physical_floor_inputs: "
                f"reported={floor:.9g}, expected={expected_floor_ms:.9g}"
            )
        if latency < floor:
            physical_floor_violations.append(
                {"phase": phase, "window_size": window, "latency_ms": latency, "floor_ms": floor}
            )
        phases.add(phase)
        key = tuple(
            row.get(field)
            for field in (
                "phase",
                "kernel_source",
                "num_heads",
                "num_key_value_heads",
                "head_dim",
                "batch_size",
                "sequence",
                "attn_dtype",
                "kv_cache_dtype",
            )
        )
        if window in grouped[key]:
            raise ValueError(f"duplicate attention fidelity row for key={key!r}, window={window}")
        grouped[key][window] = row

    if physical_floor_violations:
        raise ValueError(f"physical floor violation: {physical_floor_violations}")
    if phases != {"context", "generation"}:
        raise ValueError(f"attention fidelity must cover both phases, got {sorted(phases)}")

    pairs = []
    trend_groups: dict[tuple, list[dict]] = defaultdict(list)
    significant_window_slowdowns = []
    effective_window_discrimination_violations = []
    for key, windows in grouped.items():
        if set(windows) != {0, 512}:
            raise ValueError(f"attention fidelity pair missing window for key={key!r}")
        full = windows[0]
        windowed = windows[512]
        full_median, full_variance = _median_and_variance(full["samples_ms"])
        window_median, window_variance = _median_and_variance(windowed["samples_ms"])
        sample_count_full = len(full["samples_ms"])
        sample_count_window = len(windowed["samples_ms"])
        standard_error = math.sqrt(full_variance / sample_count_full + window_variance / sample_count_window)
        try:
            sequence_value = float(full["sequence"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"sequence must be a positive integer, got {full.get('sequence')!r}") from exc
        if not math.isfinite(sequence_value) or sequence_value <= 0 or not sequence_value.is_integer():
            raise ValueError(f"sequence must be a positive integer, got {full.get('sequence')!r}")
        sequence = int(sequence_value)
        significant_delta = window_median - full_median
        if sequence > _LONG_CONTEXT_MULTIPLIER * _WINDOW_SIZE and significant_delta > 1.96 * standard_error:
            significant_window_slowdowns.append(
                {
                    "key": key,
                    "window0_median_ms": full_median,
                    "window512_median_ms": window_median,
                    "delta_ms": significant_delta,
                    "standard_error_ms": standard_error,
                }
            )
        effective_window = 512.0 * full_median / window_median
        if sequence > _LONG_CONTEXT_MULTIPLIER * _WINDOW_SIZE and effective_window <= _WINDOW_SIZE:
            effective_window_discrimination_violations.append(
                {
                    "key": key,
                    "sequence": sequence,
                    "effective_window": effective_window,
                }
            )
        pair = {
            "key": key,
            "phase": key[0],
            "sequence": sequence,
            "window0_median_ms": full_median,
            "window512_median_ms": window_median,
            "effective_window": effective_window,
            "standard_error_ms": standard_error,
            "effective_window_standard_error": (
                effective_window
                * math.sqrt(
                    (math.sqrt(full_variance / sample_count_full) / full_median) ** 2
                    + (math.sqrt(window_variance / sample_count_window) / window_median) ** 2
                )
            ),
        }
        pairs.append(pair)
        trend_key = key[:6] + key[7:]
        trend_groups[trend_key].append(pair)

    if significant_window_slowdowns:
        raise ValueError(f"window=512 slower than window=0: {significant_window_slowdowns}")
    if effective_window_discrimination_violations:
        raise ValueError(
            f"window=512 did not discriminate beyond the fixed window: {effective_window_discrimination_violations}"
        )
    effective_window_plateaus = []
    effective_window_trend_violations = []
    for trend_key, trend_pairs in trend_groups.items():
        long_pairs = sorted(
            (pair for pair in trend_pairs if pair["sequence"] > _LONG_CONTEXT_MULTIPLIER * _WINDOW_SIZE),
            key=lambda pair: pair["sequence"],
        )
        if len(long_pairs) < 2:
            raise ValueError(
                "attention fidelity requires at least two sequence > 2 * window points per matched group: "
                f"group={trend_key!r}, sequences={[pair['sequence'] for pair in long_pairs]!r}"
            )
        for previous, current in pairwise(long_pairs):
            if current["sequence"] == previous["sequence"]:
                raise ValueError(f"duplicate sequence in effective-window trend group={trend_key!r}")
            delta = current["effective_window"] - previous["effective_window"]
            standard_error = math.sqrt(
                previous["effective_window_standard_error"] ** 2 + current["effective_window_standard_error"] ** 2
            )
            trend = {
                "group": trend_key,
                "previous_sequence": previous["sequence"],
                "current_sequence": current["sequence"],
                "previous_effective_window": previous["effective_window"],
                "current_effective_window": current["effective_window"],
                "delta": delta,
                "standard_error": standard_error,
            }
            if delta < -_TREND_Z * standard_error:
                effective_window_trend_violations.append(trend)
            elif abs(delta) <= _TREND_Z * standard_error:
                effective_window_plateaus.append(trend)
    if effective_window_trend_violations:
        raise ValueError(f"effective_window trend violations: {effective_window_trend_violations}")
    if effective_window_plateaus:
        raise ValueError(f"effective_window plateau: {effective_window_plateaus}")
    return {
        "phases": sorted(phases),
        "matched_pairs": len(pairs),
        "pairs": pairs,
        "physical_floor_violations": physical_floor_violations,
        "significant_window_slowdowns": significant_window_slowdowns,
        "effective_window_plateaus": effective_window_plateaus,
        "effective_window_trend_violations": effective_window_trend_violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-rows", type=Path)
    parser.add_argument("--image-reference", required=True)
    parser.add_argument("--image-manifest-digest", required=True)
    parser.add_argument("--image-config-digest", required=True)
    args = parser.parse_args()
    result = {
        "canonical_collection": False,
        "diagnostic_only": False,
        "image": validate_image_provenance(
            args.image_reference,
            args.image_manifest_digest,
            args.image_config_digest,
        ),
        "runtime": collect_runtime_facts(),
    }
    if args.attention_rows is not None:
        rows = json.loads(args.attention_rows.read_text(encoding="utf-8"))
        result["attention_fidelity"] = summarize_attention_fidelity(
            rows,
            required_profile=STEP4_SWA_PROFILE,
        )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": "validated"}, sort_keys=True))


if __name__ == "__main__":
    main()
