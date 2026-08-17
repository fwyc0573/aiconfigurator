from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST_SCRIPT = ROOT / "run_b300_single_smoke.sh"
REMOTE_SCRIPT = ROOT / "remote_b300_single_smoke.sh"


def _text(path: Path) -> str:
    assert path.is_file(), f"missing required script: {path}"
    return path.read_text()


def test_host_wrapper_pins_resources_source_and_mounts() -> None:
    script = _text(HOST_SCRIPT)
    assert "607d1641ee3fec43653fca510d717725828890c2" in script
    assert "c820e5ae1" in script
    assert "b300_train_infra" in script
    assert 'POSITIVE_TAGS="${POSITIVE_TAGS:-B300}"' in script
    assert "--gpu 1" in script
    assert "--memory 300000" in script
    assert 'SERVING_PORT="${SERVING_PORT:-8000}"' in script
    assert '-e PORT_AUTO0="${SERVING_PORT}"' in script
    assert "--auto-port-num" not in script
    assert "--host-network" not in script
    assert "qy1-pt" not in script
    assert "step2-alignment-jfs" not in script
    assert "step4pro_smoke_14l_dummy" in script
    assert "--entrypoint /bin/bash" in script
    assert '-- -lc "${worker_command}"' in script
    assert 'if [[ "${VLLM_ALL2ALL_BACKEND}" != "allgather_reducescatter" ]]; then' in script
    assert 'if [[ "${VLLM_ENABLE_SEQUENCE_PARALLEL}" != "0" ]]; then' in script


def test_host_wrapper_builds_identity_payload_and_recovers_evidence() -> None:
    script = _text(HOST_SCRIPT)
    assert "image_to_pinned_vllm.patch.gz" in script
    assert "pinned_vllm_manifest.sha256" in script
    assert "image_base_changed_manifest.tsv" in script
    assert "PAYLOAD_SOURCE_ROOT=" in script
    assert "IDENTITY_PAYLOAD_ROOT=" in script
    assert "pack-objects" not in script
    assert 'tar cf - -C "${PAYLOAD_ROOT}" .' in script
    assert "REMOTE_EXEC_PID=$!" in script
    assert "tar cf - -C /home" in script
    assert '>"${REMOTE_EVIDENCE_TAR}"' in script
    assert "brainctl delete rjob" in script
    assert "rjob.brainpp.cn/rjob-name=${RJOB_NAME}" in script
    assert "setsid sudo -n systemd-run" in script
    assert "timeout --signal=TERM" in script
    assert 'CONTROL_MEMORY_MAX="${CONTROL_MEMORY_MAX:-2G}"' in script
    assert '-p MemoryMax="${CONTROL_MEMORY_MAX}"' in script
    assert "b300_runtime_contract.sh" in script
    assert "cleanup_inventory_is_empty" in script
    assert "assert_runtime_log_clean" in script
    assert "kill -KILL" in script
    assert "remote_result_ready" in script
    assert "\n:\n" not in script
    assert script.count('-- -lc "${worker_command}"') == 1
    assert 'exec -i "replica/${REPLICA}"' in script
    assert script.index("brainctl delete rjob") < script.index('kill -TERM -- "-${process_id}"')


def test_remote_script_proves_pinned_runtime_source() -> None:
    script = _text(REMOTE_SCRIPT)
    assert "0.19.0.post20.dev26+gc820e5ae1" in script
    assert "sha256sum -c" in script
    assert 'git -C "${RUNTIME_REPO}" rev-parse HEAD' in script
    assert 'git -C "${RUNTIME_REPO}" cat-file -t HEAD' in script
    assert "inspect.getsourcefile(step4pro)" in script
    assert "inspect.getsourcefile(optimus_fa4)" in script
    assert "import vllm._C" in script
    assert "Step4ProForCausalLM" in script
    assert "path.read_bytes()" not in script
    assert "<<'PY' |\n" not in script
    assert "GiB(?: memory)? and" in script
    assert "optimus_jit_quant_overlay.patch" in script
    assert "expected_call_count = 5" in script
    assert "optimus_cutedsl.group_quant_fp8" in script
    assert "_optimus_jit_per_token_group_quant_fp8" in script
    assert 'scale_format="sm100_1d1d"' in script
    assert "ep_gather_block_overlay.patch" in script
    assert "optimus_triton_gather_overlay.patch" in script
    assert "hidden_size & -hidden_size" in script
    assert (
        'PINNED_GPU_MODEL_RUNNER_SHA256="298a43a69f3b5b43bdbb753b3cee642933a0dbd71368dfbf0271dba1fce32bcb"'
    ) in script
    assert "model_forward_complete_overlay.patch" in script
    assert "MODEL_FORWARD_COMPLETE" in script
    assert "synchronize = current_platform.synchronize" in script
    assert "synchronize()" in script


def test_remote_script_preserves_single_gpu_recipe_and_provider_gates() -> None:
    script = _text(REMOTE_SCRIPT)
    for marker in (
        "VLLM_KV_CACHE_LAYOUT=NHD",
        "OPTIMUS_MUST_LOAD_LIB=1",
        "VLLM_USE_OPTIMUS_MOE=1",
        "VLLM_USE_DEEP_GEMM_E8M0=1",
        "OPTIMUS_TRITON_DRIVER_STRICT_SIGNATURE=1",
        "step-optimus",
        "3.23.24",
        'for package in ("deep_gemm", "torch"):',
        "RMSNorm_forward",
        "ModuleSpec",
        "sys.modules.pop(alias)",
        "--block-size 128",
        "--enable-expert-parallel",
        "--enforce-eager",
        "--load-format dummy",
        "--skip-tokenizer-init",
        "--profiler-config.profiler=torch",
        "--profiler-config.torch_profiler_record_shapes=true",
        "--profiler-config.torch_profiler_with_stack=true",
        "/start_profile",
        "/stop_profile",
        "Optimus FA4 actual forward:",
        "Using OPTIMUS_FP8 Fp8 MoE backend",
        "OptimusFp8Experts uses legacy Optimus DeepGemm",
        "BATCH_TOKEN_IDS=PASS",
        "--data-parallel-size",
        "allgather_reducescatter",
        'VLLM_ENABLE_SEQUENCE_PARALLEL="${VLLM_ENABLE_SEQUENCE_PARALLEL:-0}"',
        "--all2all-backend ${VLLM_ALL2ALL_BACKEND}",
        "Using AgRsAll2AllManager all2all manager",
        "validate_distributed_all2all_runtime",
        "runtime_all2all_backend",
        "runtime_all2all_manager",
        "sequence_parallel",
        "agrs_manager_marker_count",
        "deepep_manager_marker_count",
        "auto_backend_selection_marker_count",
    ):
        assert marker in script
    assert "deepep_high_throughput" not in script
    assert 'for package in ("deep_gemm", "deep_ep", "torch"):' not in script
    assert 'grep -Ec "Using DeepEP[A-Za-z0-9_]*All2AllManager"' in script
    assert "Unexpected Step MoE automatic backend selection" in script
    assert "backend=HT" not in script
    assert "--tokenizer" not in script
    assert "COUNT_1_TO_100" not in script
    assert "/v1/chat/completions" not in script
    assert "Attention provider fallback detected" not in script
    assert 'ENABLE_PROFILER="${ENABLE_PROFILER:-0}"' in _text(HOST_SCRIPT)
    assert "if (( ENABLE_PROFILER == 1 )); then" in script
    assert '"max_tokens":4' in script


def test_remote_script_records_numeric_metrics_and_stops_server() -> None:
    script = _text(REMOTE_SCRIPT)
    for metric in (
        "worker_started_epoch",
        "server_started_epoch",
        "health_ready_seconds",
        "gpu_memory_before_load_mib",
        "gpu_memory_after_load_mib",
        "request_latency_seconds",
        "concurrent_wall_seconds",
        "prompt_tokens",
        "completion_tokens",
    ):
        assert metric in script
    assert 'kill "${SERVER_PID}"' in script
    assert 'wait "${SERVER_PID}"' in script


def test_remote_script_holds_evidence_until_host_collects_it() -> None:
    script = _text(REMOTE_SCRIPT)
    assert "EVIDENCE_HOLD_SECONDS" in script
    assert 'sleep "${EVIDENCE_HOLD_SECONDS}"' in script
    host_script = _text(HOST_SCRIPT)
    assert "REMOTE_RESULT_TIMEOUT_SECONDS" in host_script


def test_remote_script_uses_headless_mode_for_nonzero_dp_start_rank() -> None:
    script = _text(REMOTE_SCRIPT)
    for marker in (
        "HEADLESS_MODE=0",
        "HEADLESS_ARGS=",
        "DATA_PARALLEL_START_RANK > 0",
        'HEADLESS_ARGS="--headless --api-server-count 0"',
        'touch "${REMOTE_VALIDATION_READY_FILE}"',
        "hold_distributed_runtime_for_host_cleanup",
        "HEADLESS_RUNTIME=PASS",
        "DISTRIBUTED_RUNTIME_VALIDATION=PASS",
        "MODEL_FORWARD_COMPLETE.*batch=real",
    ):
        assert marker in script
    assert "headless_server_exit_status" not in script
    assert "await_coordinated_shutdown" not in script
    assert "COORDINATED_SHUTDOWN" not in script
    assert 'wait_for_server_log_marker \\\n        "FORWARD_CONTEXT.*batch=real"' not in script


def test_ep_gather_block_rule_preserves_exact_dimension_coverage() -> None:
    expected = {
        128: 128,
        512: 512,
        896: 128,
        1024: 1024,
        1792: 256,
        2048: 1024,
        3072: 1024,
        3584: 512,
        7168: 1024,
    }
    for hidden_size, expected_block in expected.items():
        block = min(1024, hidden_size & -hidden_size)
        assert block == expected_block
        assert block & (block - 1) == 0
        assert hidden_size % block == 0
        covered = [
            dimension
            for block_index in range(hidden_size // block)
            for dimension in range(block_index * block, (block_index + 1) * block)
        ]
        assert covered == list(range(hidden_size))
