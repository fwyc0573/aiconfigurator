"""Regression tests for vLLM GEMM execution-mode policy."""

import ast
from pathlib import Path

import pytest


@pytest.fixture
def gemm_graph_policy():
    source_path = Path(__file__).resolve().parents[3] / "collector/vllm/collect_gemm.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_use_cuda_graph_for_gemm"
    )
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["_use_cuda_graph_for_gemm"]


@pytest.mark.unit
def test_vllm_019_fp8_gemm_is_explicitly_eager(gemm_graph_policy):
    assert gemm_graph_policy("fp8", "0.19.0") is False


@pytest.mark.unit
def test_vllm_019_bf16_gemm_keeps_cuda_graph(gemm_graph_policy):
    assert gemm_graph_policy("bfloat16", "0.19.0") is True


@pytest.mark.unit
def test_vllm_019_fp8_block_gemm_is_eager(gemm_graph_policy):
    assert gemm_graph_policy("fp8_block", "0.19.0") is False


@pytest.mark.unit
def test_other_vllm_versions_keep_existing_fp8_policy(gemm_graph_policy):
    assert gemm_graph_policy("fp8", "0.20.0") is True
