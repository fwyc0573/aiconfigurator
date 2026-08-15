"""RED tests for the pinned vLLM Step4-Pro-Latest MTP-off contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import aiconfigurator.sdk.operations as ops
from aiconfigurator.sdk import common, config, models, utils

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
MODEL_ID = "stepfun-ai/Step4-Pro-Latest"
LEGACY_MODEL_ID = "stepfun-ai/Step4-Pro-V4"
MANIFEST_PATH = (
    ROOT
    / "task_memory"
    / "task_2026-08-13_step4_pro_latest_bcard_ops_collection_simulation"
    / "step4_pro_latest_shape_manifest.reconstructed.json"
)
FULL_LAYER_IDS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 77)
DENSE_LAYER_IDS = (0, 1)
LATENT_MOE_LAYER_IDS = tuple(range(2, 78))


@pytest.fixture(autouse=True)
def _clear_model_config_caches():
    """Keep cached model discovery isolated between tests."""
    utils._load_model_config_from_model_path.cache_clear()
    utils.get_model_config_from_model_path.cache_clear()
    yield
    utils._load_model_config_from_model_path.cache_clear()
    utils.get_model_config_from_model_path.cache_clear()


@pytest.fixture(scope="module")
def manifest() -> dict:
    """Load the reconstructed pinned shape contract."""
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _build_latest_model(
    *,
    nextn: int = 0,
    tp_size: int = 1,
    attention_dp_size: int = 1,
    moe_ep_size: int = 1,
):
    """Build the Latest model through the public model registry."""
    return models.get_model(
        MODEL_ID,
        config.ModelConfig(
            tp_size=tp_size,
            pp_size=1,
            attention_dp_size=attention_dp_size,
            moe_tp_size=1,
            moe_ep_size=moe_ep_size,
            nextn=nextn,
            nextn_accept_rates=[0.85, 0.0, 0.0, 0.0, 0.0],
        ),
        backend_name="vllm",
    )


def _walk_operations(operation_list):
    """Yield top-level operations and nested overlap groups in execution order."""
    for operation in operation_list:
        yield operation
        if isinstance(operation, ops.OverlapOp):
            yield from _walk_operations(operation._group_a)
            yield from _walk_operations(operation._group_b)


def _operations_by_name(operation_list) -> dict[str, object]:
    """Index a graph recursively and reject duplicate operation names."""
    indexed = {}
    for operation in _walk_operations(operation_list):
        assert operation._name not in indexed, f"duplicate operation name: {operation._name}"
        indexed[operation._name] = operation
    return indexed


def _layer_operations(model, phase: str, layer_id: int) -> list:
    """Return one complete decoder layer in execution order."""
    operations = model.context_ops if phase == "context" else model.generation_ops
    prefix = f"{phase}_layer_{layer_id:03d}_"
    return [operation for operation in _walk_operations(operations) if operation._name.startswith(prefix)]


def _assert_required_subsequence(actual: list[str], expected: list[str]) -> None:
    """Require every expected item in order."""
    cursor = iter(actual)
    for expected_name in expected:
        assert any(actual_name == expected_name for actual_name in cursor), (
            f"missing or out-of-order operation {expected_name!r}; actual order: {actual}"
        )


def _assert_gemm_shape(operation, *, n: int, k: int) -> None:
    assert operation._n == n
    assert operation._k == k


def _quant_mode_name(operation) -> str | None:
    quant_mode = getattr(operation, "_quant_mode", None)
    value = getattr(quant_mode, "value", quant_mode)
    return getattr(value, "name", None)


def _suffixes(operations, prefix: str) -> list[str]:
    """Strip one layer prefix from operation names."""
    return [operation._name.removeprefix(prefix) for operation in operations]


def test_reconstructed_manifest_is_the_pinned_mtp_off_contract(manifest):
    """The local reconstruction records only values fixed by approved sources."""
    assert manifest["reconstruction"]["status"] == "reconstructed"
    assert manifest["reconstruction"]["original_manifest_available"] is False
    assert manifest["model"] == {
        "model_id": MODEL_ID,
        "architecture": "Step4ProForCausalLM",
        "pinned_vllm_commit": "607d1641ee3fec43653fca510d717725828890c2",
        "hidden_size": 7168,
        "num_hidden_layers": 78,
        "vocab_size": 128896,
        "num_nextn_predict_layers": 0,
    }
    assert tuple(manifest["layers"]["full_attention_ids"]) == FULL_LAYER_IDS
    assert manifest["layers"]["swa_attention_count"] == 58
    assert tuple(manifest["layers"]["dense_ffn_ids"]) == DENSE_LAYER_IDS
    assert manifest["layers"]["latent_moe"] == {
        "first_layer_id": 2,
        "last_layer_id": 77,
        "count": 76,
    }
    assert manifest["attention"]["full_mfa"]["required_op_order"][8] == "inverse_rope"
    assert manifest["attention"]["full_mfa"]["attention_provider"] == "optimus_fa4"
    assert manifest["attention"]["full_mfa"]["grouped_wo_a"]["provider"] == "vllm_step4pro_torch_einsum"
    assert tuple(manifest["attention"]["swa_gqa"]["normalizes"]) == ("q", "k", "v")
    assert manifest["ffn"]["dense"]["activation"] == "situ_glu"
    assert manifest["ffn"]["latent_moe"]["router"]["provider"] == "vllm.optimus_matmul_fp32"
    assert manifest["communication"]["expert_parallel"]["provider"] == "vllm_deepep_high_throughput"
    assert manifest["kv_cache"] == {
        "requested_dtype": "auto",
        "resolved_dtype": "bfloat16",
        "page_size": 128,
        "full_mfa_elements_per_token_per_layer": 512,
        "swa_gqa_elements_per_token_per_layer": 2048,
        "full_mfa_kv_storage_alias": True,
    }
    assert manifest["precision"]["router"] == "float32"
    assert manifest["mtp"]["enabled"] is False
    assert manifest["mtp"]["layers"] == 0


def test_latest_has_independent_step4pro_identity():
    """Latest must not alias the legacy Step4ForCausalLM model identity."""
    assert common.ARCHITECTURE_TO_MODEL_FAMILY.get("Step4ProForCausalLM") is not None
    assert MODEL_ID in common.DefaultHFModels

    latest_raw = utils._load_pre_downloaded_hf_config(MODEL_ID)
    legacy_raw = utils._load_pre_downloaded_hf_config(LEGACY_MODEL_ID)
    assert latest_raw["architectures"] == ["Step4ProForCausalLM"]
    assert legacy_raw["architectures"] == ["Step4ForCausalLM"]

    model = _build_latest_model()
    assert model.model_path == MODEL_ID
    assert model.architecture == "Step4ProForCausalLM"
    assert model.extra_params.__class__.__name__ == "Step4ProLatestConfig"


def test_latest_layer_map_has_78_layers_with_dense_then_latent_moe(manifest):
    """The parsed config preserves the exact heterogeneous trunk layout."""
    info = utils.get_model_config_from_model_path(MODEL_ID)
    extra = info["extra_params"]
    records = tuple((layer.layer_id, layer.attention_type, layer.ffn_type) for layer in extra.layers)
    expected = tuple(
        (
            layer_id,
            "full" if layer_id in FULL_LAYER_IDS else "swa",
            "dense" if layer_id in DENSE_LAYER_IDS else "latent_moe",
        )
        for layer_id in range(78)
    )

    assert info["architecture"] == "Step4ProForCausalLM"
    assert info["layers"] == 78
    assert info["hidden_size"] == 7168
    assert info["num_experts"] == 896
    assert info["topk"] == 16
    assert info["moe_inter_size"] == 3584
    assert records == expected
    assert tuple(layer_id for layer_id, _, ffn_type in records if ffn_type == "dense") == DENSE_LAYER_IDS
    assert tuple(layer_id for layer_id, _, ffn_type in records if ffn_type == "latent_moe") == LATENT_MOE_LAYER_IDS
    assert sum(attention_type == "full" for _, attention_type, _ in records) == 20
    assert sum(attention_type == "swa" for _, attention_type, _ in records) == 58
    assert manifest["ffn"]["dense"]["intermediate_size"] == extra.dense_inter_size
    assert manifest["ffn"]["latent_moe"]["latent_hidden_size"] == extra.latent_moe_dim


def test_latest_kv_capacity_uses_hybrid_curve():
    """Latest capacity inversion must account for the SWA window saturation."""
    model = _build_latest_model()
    budget = model.get_kvcache_bytes_per_sequence(513)

    assert model.get_kvcache_max_tokens(budget) == 513


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_latest_full_mfa_key_order_and_dimensions(phase, manifest):
    """Full layers expose the pinned low-rank shared-KV and grouped-output path."""
    model = _build_latest_model()
    full = manifest["attention"]["full_mfa"]
    prefix = f"{phase}_layer_003_full_"
    layer_ops = [
        operation for operation in _layer_operations(model, phase, layer_id=3) if operation._name.startswith(prefix)
    ]
    suffixes = _suffixes(layer_ops, prefix)
    _assert_required_subsequence(suffixes, full["required_op_order"])
    by_name = {operation._name.removeprefix(prefix): operation for operation in layer_ops}

    for name, shape in full["gemm_shapes"].items():
        _assert_gemm_shape(by_name[name], **shape)
    for name in ("wq_a_gemm", "wq_b_gemm", "wkv_gemm", "head_gate_gemm", "wo_b_gemm"):
        assert _quant_mode_name(by_name[name]) == "bfloat16"

    attention = by_name["attention"]
    assert attention._n == 64
    assert attention._n_kv == 1
    assert attention._head_size == 512
    assert attention._window_size == 0
    assert attention._provider == "optimus_fa4"
    assert attention._kv_storage_alias is True
    assert attention._page_size == 128

    grouped_wo_a = by_name["wo_a_grouped_gemm"]
    assert grouped_wo_a.__class__.__name__ == "GroupedGEMM"
    assert grouped_wo_a._provider == "vllm_step4pro_torch_einsum"
    assert grouped_wo_a._groups == 8
    assert grouped_wo_a._n == 1024
    assert grouped_wo_a._k == 4096
    assert _quant_mode_name(grouped_wo_a) == "bfloat16"
    assert suffixes.count("inverse_rope") == 1


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_latest_swa_gqa_key_order_and_dimensions(phase, manifest):
    """SWA layers use packed QKV GQA with the pinned 512-token window."""
    model = _build_latest_model()
    swa = manifest["attention"]["swa_gqa"]
    prefix = f"{phase}_layer_002_swa_"
    layer_ops = [
        operation for operation in _layer_operations(model, phase, layer_id=2) if operation._name.startswith(prefix)
    ]
    suffixes = _suffixes(layer_ops, prefix)
    _assert_required_subsequence(suffixes, swa["required_op_order"])
    by_name = {operation._name.removeprefix(prefix): operation for operation in layer_ops}

    for name, shape in swa["gemm_shapes"].items():
        _assert_gemm_shape(by_name[name], **shape)
        assert _quant_mode_name(by_name[name]) == "bfloat16"

    attention = by_name["attention"]
    assert attention._n == 128
    assert attention._n_kv == 8
    assert attention._head_size == 128
    assert attention._window_size == 512
    assert attention._provider == "vllm_native_sliding_gqa"
    qkv_norm_rope = by_name["qkv_norm_rope"]
    assert qkv_norm_rope._normalized_tensors == ("q", "k", "v")
    assert qkv_norm_rope._provider == "vllm_step4pro_qkv_norm_rope"


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_latest_dense_ffn_uses_pinned_situ_glu_order_and_shapes(phase, manifest):
    """Dense layers preserve the pinned norm, merged projection, SiTU-GLU, and residual order."""
    model = _build_latest_model()
    dense = manifest["ffn"]["dense"]
    prefix = f"{phase}_layer_000_dense_"
    layer_ops = [
        operation for operation in _layer_operations(model, phase, layer_id=0) if operation._name.startswith(prefix)
    ]
    suffixes = _suffixes(layer_ops, prefix)
    _assert_required_subsequence(suffixes, dense["required_op_order"])
    by_name = {operation._name.removeprefix(prefix): operation for operation in layer_ops}

    _assert_gemm_shape(by_name["gate_up_gemm"], **dense["gate_up_gemm"])
    _assert_gemm_shape(by_name["down_gemm"], **dense["down_gemm"])
    assert _quant_mode_name(by_name["gate_up_gemm"]) == "bfloat16"
    assert _quant_mode_name(by_name["down_gemm"]) == "bfloat16"
    assert (by_name["situ_glu"]._dim_in, by_name["situ_glu"]._dim_out) == (52224, 26112)
    assert not any("shared" in suffix or "moe" in suffix for suffix in suffixes)


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_latest_latent_moe_is_serial_and_uses_exact_provider_identities(phase, manifest):
    """Latent MoE must follow the pinned routed, shared, post-projection, and merge order."""
    model = _build_latest_model()
    latent = manifest["ffn"]["latent_moe"]
    prefix = f"{phase}_layer_002_latent_moe_"
    layer_ops = [
        operation for operation in _layer_operations(model, phase, layer_id=2) if operation._name.startswith(prefix)
    ]
    suffixes = _suffixes(layer_ops, prefix)
    _assert_required_subsequence(suffixes, latent["required_op_order"])
    by_name = {operation._name.removeprefix(prefix): operation for operation in layer_ops}

    assert not any(isinstance(operation, ops.OverlapOp) for operation in _layer_operations(model, phase, 2))

    router = by_name["router_gemm"]
    router_shape = latent["router"]
    _assert_gemm_shape(router, n=router_shape["n"], k=router_shape["k"])
    assert router.__class__.__name__ == "FP32OutputGEMM"
    assert router._provider == "vllm.optimus_matmul_fp32"
    assert router._weight_dtype == "bfloat16"
    assert router._output_dtype == "float32"

    pre_proj = by_name["pre_proj"]
    post_proj = by_name["post_proj"]
    _assert_gemm_shape(pre_proj, n=3584, k=7168)
    _assert_gemm_shape(post_proj, n=7168, k=3584)
    assert _quant_mode_name(pre_proj) == "bfloat16"
    assert _quant_mode_name(post_proj) == "bfloat16"

    dispatch = by_name["dispatch"]
    combine = by_name["combine"]
    assert dispatch._provider == "vllm_deepep_high_throughput"
    assert dispatch._operation == "dispatch"
    assert combine._provider == "vllm_deepep_high_throughput"
    assert combine._operation == "combine"

    routed = by_name["experts"]
    assert routed._hidden_size == 3584
    assert routed._inter_size == 3584
    assert routed._num_experts == 896
    assert routed._topk == 16
    assert routed._provider == "optimus_fp8_moe"
    assert _quant_mode_name(routed) == "fp8_block"

    _assert_gemm_shape(by_name["shared_gate_up_gemm"], n=7168, k=7168)
    assert _quant_mode_name(by_name["shared_gate_up_gemm"]) == "bfloat16"
    assert (by_name["shared_situ_glu"]._dim_in, by_name["shared_situ_glu"]._dim_out) == (7168, 3584)
    _assert_gemm_shape(by_name["shared_down_gemm"], n=7168, k=3584)
    assert _quant_mode_name(by_name["shared_down_gemm"]) == "bfloat16"


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_latest_graph_is_complete_and_interleaves_attention_with_each_layer_ffn(phase):
    """The graph must preserve all 78 decoder-layer boundaries instead of aggregating by op family."""
    model = _build_latest_model()
    operations = model.context_ops if phase == "context" else model.generation_ops
    names = [operation._name for operation in operations]

    assert names[0] == f"{phase}_embedding"
    assert names[-2:] == [f"{phase}_final_norm", f"{phase}_logits_gemm"]
    assert not any(isinstance(operation, ops.OverlapOp) for operation in operations)

    previous_end = 0
    for layer_id in range(78):
        layer_ops = _layer_operations(model, phase, layer_id)
        layer_names = [operation._name for operation in layer_ops]
        attention_type = "full" if layer_id in FULL_LAYER_IDS else "swa"
        ffn_type = "dense" if layer_id in DENSE_LAYER_IDS else "latent_moe"
        assert layer_names[0] == f"{phase}_layer_{layer_id:03d}_{attention_type}_attn_norm"
        assert f"{phase}_layer_{layer_id:03d}_attention_residual_add" in layer_names
        assert layer_names[-1] == f"{phase}_layer_{layer_id:03d}_{ffn_type}_ffn_residual_add"
        start = names.index(layer_names[0])
        end = names.index(layer_names[-1])
        assert previous_end < start < end
        previous_end = end


@pytest.mark.parametrize("ep_size", [16, 32])
def test_latest_deepep_dispatch_and_combine_keys_remain_distinct(ep_size):
    """EP16 and EP32 must retain separate dispatch/combine provider identities."""
    model = _build_latest_model(attention_dp_size=ep_size, moe_ep_size=ep_size)
    layer = _operations_by_name(_layer_operations(model, "generation", 2))
    dispatch = layer["generation_layer_002_latent_moe_dispatch"]
    combine = layer["generation_layer_002_latent_moe_combine"]

    assert dispatch._moe_ep_size == ep_size
    assert combine._moe_ep_size == ep_size
    assert dispatch._operation == "dispatch"
    assert combine._operation == "combine"
    assert dispatch._persisted_key() != combine._persisted_key()


def test_latest_kv_cache_reports_logical_and_page_allocated_bytes(manifest):
    """KV accounting must preserve Full-MFA aliasing, SWA retention, and 128-token page padding."""
    model = _build_latest_model()
    kv = manifest["kv_cache"]
    bytes_per_element = 2

    expected_logical_513 = (
        20 * 513 * kv["full_mfa_elements_per_token_per_layer"] + 58 * 512 * kv["swa_gqa_elements_per_token_per_layer"]
    ) * bytes_per_element
    expected_allocated_513 = (
        20 * 640 * kv["full_mfa_elements_per_token_per_layer"] + 58 * 512 * kv["swa_gqa_elements_per_token_per_layer"]
    ) * bytes_per_element

    assert model.kv_cache_requested_dtype == "auto"
    assert model.kv_cache_resolved_dtype == "bfloat16"
    assert model.kv_cache_page_size == 128
    assert model.config.gemm_quant_mode == common.GEMMQuantMode.bfloat16
    assert model.config.fmha_quant_mode == common.FMHAQuantMode.bfloat16
    assert model.config.kvcache_quant_mode == common.KVCacheQuantMode.bfloat16
    assert model.get_kvcache_bytes_per_sequence(513) == expected_logical_513
    assert model.get_kvcache_allocated_bytes_per_sequence(513) == expected_allocated_513


def test_latest_has_no_mtp1_graph_or_generic_nextn_scaling():
    """Pinned Latest is MTP-off and must reject generic Step3p5-style nextn."""
    raw = utils._load_pre_downloaded_hf_config(MODEL_ID)
    assert raw["num_nextn_predict_layers"] == 0

    model = _build_latest_model(nextn=0)
    operation_names = [operation._name.lower() for operation in _walk_operations(model.generation_ops)]
    assert not any(marker in name for name in operation_names for marker in ("mtp", "nextn", "predictor"))

    with pytest.raises((ValueError, NotImplementedError), match=r"(?i)(mtp1|nextn|multi-token)"):
        _build_latest_model(nextn=1)

    with pytest.raises((ValueError, NotImplementedError), match=r"(?i)(full mfa|tp.?1|tensor parallel)"):
        _build_latest_model(tp_size=2)
