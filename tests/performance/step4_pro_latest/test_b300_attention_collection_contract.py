"""Static contracts for the Step4-Pro-Latest B300 Attention collector."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent
HOST_SCRIPT = ROOT / "run_b300_attention_collection.sh"
REMOTE_SCRIPT = ROOT / "remote_b300_attention_collection.sh"


def test_host_wrapper_uses_exact_b300_runtime_and_bounded_control_plane():
    source = HOST_SCRIPT.read_text(encoding="utf-8")

    assert "b300_train_infra" in source
    assert "--positive-tags" in source
    assert "B300" in source
    assert "MemoryMax=3G" in source
    assert "--backoff-limit 1" in source
    assert "--volume" not in source
    assert "2026-08-06-server-vllm-test-0.19.0.post20.dev26.gc820e5ae1.precompiled" in source
    assert "607d1641ee3fec43653fca510d717725828890c2" in source
    assert "aic_metadata.tar" in source


def test_remote_smoke_executes_both_pinned_attention_providers():
    source = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert "optimus_fa4" in source
    assert "vllm_native_sliding_gqa" in source
    assert "run_step4_context_attention" in source
    assert "run_step4_generation_attention" in source
    assert "VLLM_KV_CACHE_LAYOUT=NHD" in source
    assert '"physical_page_bytes": 524288' in source
    assert "kv_block_stride_bytes=524288" in source
    assert "kv_block_stride_bytes=262144" in source


def test_attention_smoke_can_target_provider_specific_query_and_prefix_points():
    host_source = HOST_SCRIPT.read_text(encoding="utf-8")
    remote_source = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert 'FULL_CONTEXT_SMOKE_TOKENS="${FULL_CONTEXT_SMOKE_TOKENS:-512}"' in host_source
    assert "export FULL_CONTEXT_SMOKE_TOKENS=" in host_source
    assert (
        'FULL_CONTEXT_SMOKE_TOTAL_TOKENS="${FULL_CONTEXT_SMOKE_TOTAL_TOKENS:-${FULL_CONTEXT_SMOKE_TOKENS}}"'
        in host_source
    )
    assert "export FULL_CONTEXT_SMOKE_TOTAL_TOKENS=" in host_source
    assert 'SWA_CONTEXT_SMOKE_TOKENS="${SWA_CONTEXT_SMOKE_TOKENS:-512}"' in host_source
    assert "export SWA_CONTEXT_SMOKE_TOKENS=" in host_source
    assert (
        'SWA_CONTEXT_SMOKE_TOTAL_TOKENS="${SWA_CONTEXT_SMOKE_TOTAL_TOKENS:-${SWA_CONTEXT_SMOKE_TOKENS}}"' in host_source
    )
    assert "export SWA_CONTEXT_SMOKE_TOTAL_TOKENS=" in host_source
    assert 'FULL_CONTEXT_SMOKE_TOKENS="${FULL_CONTEXT_SMOKE_TOKENS:-512}"' in remote_source
    assert 'int(os.environ["FULL_CONTEXT_SMOKE_TOKENS"])' in remote_source
    assert 'int(os.environ["FULL_CONTEXT_SMOKE_TOTAL_TOKENS"])' in remote_source
    assert "total_context_tokens=full_total_context_tokens" in remote_source
    assert 'SWA_CONTEXT_SMOKE_TOKENS="${SWA_CONTEXT_SMOKE_TOKENS:-512}"' in remote_source
    assert 'int(os.environ["SWA_CONTEXT_SMOKE_TOKENS"])' in remote_source
    assert 'int(os.environ["SWA_CONTEXT_SMOKE_TOTAL_TOKENS"])' in remote_source
    assert "total_context_tokens=swa_total_context_tokens" in remote_source


def test_remote_collection_keeps_resumable_outputs():
    source = REMOTE_SCRIPT.read_text(encoding="utf-8")

    assert "checkpoint" in source
    assert "step4_context_attention_perf.txt" in source
    assert "step4_generation_attention_perf.txt" in source
    assert "remote_result_ready" in source
    assert "AIC_METADATA_PATH" in source
