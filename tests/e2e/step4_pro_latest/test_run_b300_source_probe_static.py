from pathlib import Path

SCRIPT = (Path(__file__).resolve().parent / "run_b300_source_probe.sh").read_text()
REMOTE_SCRIPT = (Path(__file__).resolve().parent / "remote_b300_source_probe.sh").read_text()


def test_replica_queries_are_exact_name_and_bounded() -> None:
    assert 'RJOB_LABEL="rjob.brainpp.cn/rjob-name=${RJOB_NAME}"' in SCRIPT
    assert '-l "${RJOB_LABEL}"' in SCRIPT
    assert 'get replica -n "${NAMESPACE}"' in SCRIPT
    assert "MemoryMax=3G" in SCRIPT
    assert 'timeout --signal=TERM --kill-after=5s "${timeout_seconds}s"' in SCRIPT
    assert "if (( ${#RJOB_NAME} > 50 )); then" in SCRIPT


def test_live_launcher_runs_in_parallel_with_probe_and_cleans_up_first() -> None:
    assert "setsid sudo -n systemd-run" in SCRIPT
    assert "LAUNCH_PID=$!" in SCRIPT
    assert 'scoped "${READY_TIMEOUT_SECONDS}" /kubebrain/brainctl rjob launch' not in SCRIPT
    assert SCRIPT.index("brainctl delete rjob") < SCRIPT.index('kill -TERM -- "-${LAUNCH_PID}"')


def test_remote_repository_name_tracks_transported_payload() -> None:
    assert 'REMOTE_REPO="${REMOTE_PARENT}/pinned-vllm-source-${RJOB_NAME}"' in SCRIPT
    assert "PAYLOAD_SOURCE_ROOT=" in SCRIPT
    assert "pack-objects" not in SCRIPT
    assert 'tar cf - -C "${PAYLOAD_ROOT}" .' in SCRIPT
    assert 'tar cf - -C "$(dirname "${LOCAL_REPO}")" "$(basename "${LOCAL_REPO}")"' not in SCRIPT
    assert "identity_pack" in SCRIPT


def test_entrypoint_and_source_identity_are_fail_fast() -> None:
    assert "--entrypoint /bin/bash" in SCRIPT
    assert '-- -lc "${worker_command}"' in SCRIPT
    assert SCRIPT.count('/kubebrain/brainctl -n "${NAMESPACE}" exec -i') >= 4
    assert "remote_b300_source_probe.sh" in SCRIPT
    assert 'git -C "${RUNTIME_REPO}" rev-parse HEAD' in REMOTE_SCRIPT
    assert "inspect.getsourcefile(step4pro)" in REMOTE_SCRIPT
    assert "inspect.getsourcefile(optimus_fa4)" in REMOTE_SCRIPT
    assert "optimus_fp8_moe.py" in REMOTE_SCRIPT
    assert "sha256sum \\" in REMOTE_SCRIPT
    assert "SOURCE_PROBE=PASS" in REMOTE_SCRIPT
