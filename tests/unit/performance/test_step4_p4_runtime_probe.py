"""P4 runtime-facts and attention-fidelity harness tests."""

import pytest

from tests.performance.step4_p4_runtime_probe import (
    REQUIRED_VLLM_VERSION,
    STEP4_SWA_PROFILE,
    summarize_attention_fidelity,
    validate_image_provenance,
    validate_runtime_facts,
)

pytestmark = pytest.mark.unit


def _facts():
    return {
        "vllm_version": REQUIRED_VLLM_VERSION,
        "device_name": "NVIDIA H800 SXM",
        "compute_capability": [9, 0],
        "cuda_version": "12.8",
        "nccl_version": "2.25.1",
        "flash_attn_provider": "top_level",
        "flash_attn_module": "flash_attn",
        "flash_attn_backend": "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend",
        "flash_attn_version": "2.7.4",
        "flash_attn_provenance_sha256": "b" * 64,
        "sm_clock_mhz": 1980,
        "memory_clock_mhz": 1593,
    }


def _row(phase, window, latency, floor, samples):
    return {
        "phase": phase,
        "kernel_source": "vllm_flashinfer",
        "num_heads": 96,
        "num_key_value_heads": 12,
        "head_dim": 128,
        "batch_size": 1,
        "sequence": 4096,
        "attn_dtype": "bfloat16",
        "kv_cache_dtype": "fp8",
        "window_size": window,
        "latency_ms": latency,
        "physical_floor_ms": floor,
        "physical_floor_source": "derived_h800_peak_and_traffic",
        "physical_floor_inputs": {
            "compute_flops": 1e9,
            "tc_flops": 1e12,
            "traffic_bytes": 1e6,
            "mem_bw_bytes_per_s": 1e9,
        },
        "samples_ms": samples,
    }


def _apply_step4_swa_profile(row):
    row.update(STEP4_SWA_PROFILE)
    sequence = row["sequence"]
    window = row["window_size"]
    head_dim = STEP4_SWA_PROFILE["head_dim"]
    heads = STEP4_SWA_PROFILE["num_heads"]
    kv_heads = STEP4_SWA_PROFILE["num_key_value_heads"]
    if row["phase"] == "context":
        if window == 0:
            compute_flops = 2 * head_dim * heads * sequence * (sequence + 1)
            effective_tokens = sequence
        else:
            compute_flops = 4 * head_dim * heads * sum(min(window, i + 1) for i in range(sequence))
            effective_tokens = min(sequence, window)
        traffic_bytes = 4 * head_dim * heads * sequence + 4 * head_dim * kv_heads * effective_tokens
    else:
        effective_tokens = sequence if window == 0 else min(sequence, window)
        compute_flops = 4 * head_dim * heads * effective_tokens
        traffic_bytes = 4 * head_dim * heads + 2 * head_dim * kv_heads * effective_tokens + 2 * head_dim * kv_heads
    row["physical_floor_inputs"] = {
        "compute_flops": compute_flops,
        "tc_flops": 989_000_000_000_000,
        "traffic_bytes": traffic_bytes,
        "mem_bw_bytes_per_s": 3_350_000_000_000,
    }
    row["physical_floor_ms"] = max(
        compute_flops / 989_000_000_000_000 * 1000.0,
        traffic_bytes / 3_350_000_000_000 * 1000.0,
    )


def test_runtime_facts_require_exact_vllm_and_h800_sm90():
    validate_runtime_facts(_facts())
    bad = _facts()
    bad["vllm_version"] = "0.18.0"
    with pytest.raises(ValueError, match="vLLM version"):
        validate_runtime_facts(bad)
    bad = _facts()
    bad["compute_capability"] = [8, 0]
    with pytest.raises(ValueError, match="SM90"):
        validate_runtime_facts(bad)


def test_runtime_facts_accept_bundled_vllm_flash_attention_provider():
    facts = {
        "vllm_version": REQUIRED_VLLM_VERSION,
        "device_name": "NVIDIA H800 SXM",
        "compute_capability": [9, 0],
        "cuda_version": "12.9",
        "nccl_version": "2.27.5",
        "flash_attn_provider": "vllm_bundled",
        "flash_attn_module": "vllm.vllm_flash_attn",
        "flash_attn_backend": "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend",
        "flash_attn_version": "3",
        "flash_attn_provenance_sha256": "a" * 64,
        "sm_clock_mhz": 345,
        "memory_clock_mhz": 2619,
    }
    validate_runtime_facts(facts)


def test_runtime_facts_rejects_invalid_flash_attention_identity():
    facts = {
        "vllm_version": REQUIRED_VLLM_VERSION,
        "device_name": "NVIDIA H800 SXM",
        "compute_capability": [9, 0],
        "cuda_version": "12.9",
        "nccl_version": "2.27.5",
        "flash_attn_provider": "flashinfer",
        "flash_attn_module": "flashinfer",
        "flash_attn_backend": "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend",
        "flash_attn_version": "3",
        "flash_attn_provenance_sha256": "a" * 64,
        "sm_clock_mhz": 345,
        "memory_clock_mhz": 2619,
    }
    with pytest.raises(ValueError, match="unsupported FlashAttention provider"):
        validate_runtime_facts(facts)

    facts["flash_attn_provider"] = "vllm_bundled"
    facts["flash_attn_module"] = "vllm.vllm_flash_attn"
    facts["flash_attn_backend"] = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"
    with pytest.raises(ValueError, match="FlashAttention backend"):
        validate_runtime_facts(facts)

    facts["flash_attn_backend"] = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    facts["flash_attn_provenance_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="provenance"):
        validate_runtime_facts(facts)


def test_runtime_facts_enforce_provider_specific_module_and_version():
    facts = _facts()
    facts.update(
        {
            "flash_attn_provider": "vllm_bundled",
            "flash_attn_module": "evil.module",
            "flash_attn_version": "3",
        }
    )
    with pytest.raises(ValueError, match="module"):
        validate_runtime_facts(facts)

    facts["flash_attn_module"] = "vllm.vllm_flash_attn"
    facts["flash_attn_version"] = "2"
    with pytest.raises(ValueError, match="version"):
        validate_runtime_facts(facts)

    facts = _facts()
    facts["flash_attn_module"] = "evil.module"
    with pytest.raises(ValueError, match="module"):
        validate_runtime_facts(facts)


def test_image_provenance_requires_reference_and_sha256_digests():
    result = validate_image_provenance(
        "hub.stepfun-inc.com/stepcast/stepcast:vllm-openai-v0.19.0",
        "sha256:" + "a" * 64,
        "sha256:" + "b" * 64,
    )
    assert result["manifest_digest"] == "sha256:" + "a" * 64
    with pytest.raises(ValueError, match="image reference"):
        validate_image_provenance("", "sha256:" + "a" * 64, "sha256:" + "b" * 64)
    with pytest.raises(ValueError, match="manifest digest"):
        validate_image_provenance("image:tag", "not-a-digest", "sha256:" + "b" * 64)
    with pytest.raises(ValueError, match="config digest"):
        validate_image_provenance("image:tag", "sha256:" + "a" * 64, "sha256:ABC")


def test_attention_fidelity_reports_both_phases_and_effective_window():
    rows = [
        _row("context", 0, 10.0, 1.0, [9.9, 10.0, 10.1]),
        _row("context", 512, 2.0, 1.0, [1.9, 2.0, 2.1]),
        _row("context", 0, 40.0, 1.0, [39.9, 40.0, 40.1]),
        _row("context", 512, 4.0, 1.0, [3.9, 4.0, 4.1]),
        _row("generation", 0, 4.0, 1.0, [3.9, 4.0, 4.1]),
        _row("generation", 512, 1.5, 1.0, [1.4, 1.5, 1.6]),
        _row("generation", 0, 16.0, 1.0, [15.9, 16.0, 16.1]),
        _row("generation", 512, 2.0, 1.0, [1.9, 2.0, 2.1]),
    ]
    for row in rows[2:4]:
        row["sequence"] = 16384
    for row in rows[6:8]:
        row["sequence"] = 16384
    result = summarize_attention_fidelity(rows)
    assert set(result["phases"]) == {"context", "generation"}
    assert result["matched_pairs"] == 4
    assert result["pairs"][0]["effective_window"] > 512
    assert result["physical_floor_violations"] == []
    assert result["significant_window_slowdowns"] == []
    assert result["effective_window_plateaus"] == []


def test_attention_fidelity_allows_boundary_window_noise_before_long_context_discrimination():
    rows = []
    for phase in ("context", "generation"):
        for sequence, full_latency, windowed_latency in (
            (1024, 2.0, 2.1),
            (4096, 8.0, 2.0),
            (16384, 32.0, 2.0),
        ):
            full = _row(phase, 0, full_latency, 1.0, [full_latency - 0.01, full_latency, full_latency + 0.01])
            windowed = _row(
                phase,
                512,
                windowed_latency,
                1.0,
                [windowed_latency - 0.01, windowed_latency, windowed_latency + 0.01],
            )
            full["sequence"] = sequence
            windowed["sequence"] = sequence
            rows.extend((full, windowed))
    result = summarize_attention_fidelity(rows)
    assert result["matched_pairs"] == 6
    assert result["significant_window_slowdowns"] == []
    assert result["effective_window_plateaus"] == []


def test_attention_fidelity_requires_exact_step4_swa_profile_when_requested():
    rows = []
    for phase in ("context", "generation"):
        for sequence, full_latency, windowed_latency in ((4096, 8.0, 2.0), (16384, 32.0, 2.0)):
            for window, latency in ((0, full_latency), (512, windowed_latency)):
                row = _row(phase, window, latency, 1.0, [latency - 0.01, latency, latency + 0.01])
                row["sequence"] = sequence
                rows.append(row)
    with pytest.raises(ValueError, match="Step4 SWA profile mismatch"):
        summarize_attention_fidelity(rows, required_profile=STEP4_SWA_PROFILE)

    for row in rows:
        _apply_step4_swa_profile(row)
    result = summarize_attention_fidelity(rows, required_profile=STEP4_SWA_PROFILE)
    assert result["matched_pairs"] == 4


def test_attention_fidelity_rejects_fabricated_step4_swa_physical_inputs():
    rows = []
    for phase in ("context", "generation"):
        for sequence, full_latency, windowed_latency in ((4096, 8.0, 2.0), (16384, 32.0, 2.0)):
            for window, latency in ((0, full_latency), (512, windowed_latency)):
                row = _row(phase, window, latency, 1.0, [latency - 0.01, latency, latency + 0.01])
                row["sequence"] = sequence
                _apply_step4_swa_profile(row)
                rows.append(row)
    rows[0]["physical_floor_inputs"]["traffic_bytes"] -= 1
    rows[0]["physical_floor_ms"] = max(
        rows[0]["physical_floor_inputs"]["compute_flops"] / rows[0]["physical_floor_inputs"]["tc_flops"] * 1000.0,
        rows[0]["physical_floor_inputs"]["traffic_bytes"]
        / rows[0]["physical_floor_inputs"]["mem_bw_bytes_per_s"]
        * 1000.0,
    )
    with pytest.raises(ValueError, match="derived Step4 SWA physical inputs"):
        summarize_attention_fidelity(rows, required_profile=STEP4_SWA_PROFILE)


def test_attention_fidelity_rejects_latency_sample_median_mismatch():
    rows = []
    for phase in ("context", "generation"):
        for sequence, full_latency, windowed_latency in ((4096, 8.0, 2.0), (16384, 32.0, 2.0)):
            for window, latency in ((0, full_latency), (512, windowed_latency)):
                row = _row(phase, window, latency, 1.0, [latency - 0.01, latency, latency + 0.01])
                row["sequence"] = sequence
                rows.append(row)
    rows[0]["latency_ms"] = 9.0
    with pytest.raises(ValueError, match="latency_ms must equal samples median"):
        summarize_attention_fidelity(rows)


def test_attention_fidelity_requires_repeated_samples():
    rows = []
    for phase in ("context", "generation"):
        for sequence, full_latency, windowed_latency in ((4096, 8.0, 2.0), (16384, 32.0, 2.0)):
            for window, latency in ((0, full_latency), (512, windowed_latency)):
                row = _row(phase, window, latency, 1.0, [latency])
                row["sequence"] = sequence
                rows.append(row)
    with pytest.raises(ValueError, match="at least 3 samples"):
        summarize_attention_fidelity(rows)


def test_attention_fidelity_rejects_physical_floor_violation():
    rows = [
        _row("context", 0, 0.5, 1.0, [0.5, 0.5, 0.5]),
        _row("context", 512, 0.5, 1.0, [0.5, 0.5, 0.5]),
        _row("generation", 0, 2.0, 1.0, [2.0, 2.0, 2.0]),
        _row("generation", 512, 1.0, 1.0, [1.0, 1.0, 1.0]),
    ]
    with pytest.raises(ValueError, match="physical floor"):
        summarize_attention_fidelity(rows)


def test_attention_fidelity_rejects_significant_window_slowdown():
    rows = [
        _row("context", 0, 2.0, 1.0, [1.9, 2.0, 2.1]),
        _row("context", 512, 4.0, 1.0, [3.9, 4.0, 4.1]),
        _row("generation", 0, 2.0, 1.0, [1.9, 2.0, 2.1]),
        _row("generation", 512, 1.0, 1.0, [0.9, 1.0, 1.1]),
    ]
    for row in rows:
        row["sequence"] = 4096
    with pytest.raises(ValueError, match="window=512 slower"):
        summarize_attention_fidelity(rows)


def test_attention_fidelity_rejects_unverifiable_physical_floor():
    rows = [
        _row("context", 0, 10.0, 1.1, [9.9, 10.0, 10.1]),
        _row("context", 512, 2.0, 1.0, [1.9, 2.0, 2.1]),
        _row("generation", 0, 4.0, 1.0, [3.9, 4.0, 4.1]),
        _row("generation", 512, 1.5, 1.0, [1.4, 1.5, 1.6]),
    ]
    with pytest.raises(ValueError, match="does not match physical_floor_inputs"):
        summarize_attention_fidelity(rows)

    rows[0]["physical_floor_source"] = "reported_by_worker"
    rows[0]["physical_floor_ms"] = 1.0
    with pytest.raises(ValueError, match="physical_floor_source"):
        summarize_attention_fidelity(rows)


def test_attention_fidelity_rejects_nonfinite_samples():
    rows = [
        _row("context", 0, 10.0, 1.0, [9.9, float("nan"), 10.1]),
        _row("context", 512, 2.0, 1.0, [1.9, 2.0, 2.1]),
        _row("generation", 0, 4.0, 1.0, [3.9, 4.0, 4.1]),
        _row("generation", 512, 1.5, 1.0, [1.4, 1.5, 1.6]),
    ]
    with pytest.raises(ValueError, match="finite and positive"):
        summarize_attention_fidelity(rows)


def test_attention_fidelity_rejects_significant_effective_window_decrease():
    rows = []
    for phase in ("context", "generation"):
        for sequence, full_latency, windowed_latency in (
            (4096, 10.0, 1.0),
            (16384, 5.0, 1.0),
        ):
            full = _row(phase, 0, full_latency, 1.0, [full_latency - 0.1, full_latency, full_latency + 0.1])
            windowed = _row(
                phase,
                512,
                windowed_latency,
                1.0,
                [windowed_latency - 0.1, windowed_latency, windowed_latency + 0.1],
            )
            full["sequence"] = sequence
            windowed["sequence"] = sequence
            rows.extend((full, windowed))
    with pytest.raises(ValueError, match="effective_window trend violations"):
        summarize_attention_fidelity(rows)


def test_attention_fidelity_rejects_effective_window_plateau():
    rows = []
    for phase in ("context", "generation"):
        for sequence in (1024, 4096, 16384):
            full = _row(phase, 0, 10.0, 1.0, [9.9, 10.0, 10.1])
            windowed = _row(phase, 512, 1.0, 1.0, [0.9, 1.0, 1.1])
            full["sequence"] = sequence
            windowed["sequence"] = sequence
            rows.extend((full, windowed))
    with pytest.raises(ValueError, match="effective_window plateau"):
        summarize_attention_fidelity(rows)
