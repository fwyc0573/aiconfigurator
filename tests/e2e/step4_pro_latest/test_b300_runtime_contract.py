import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTRACT_LIB = ROOT / "b300_runtime_contract.sh"


def _run_contract(function: str, *args: str) -> subprocess.CompletedProcess[str]:
    assert CONTRACT_LIB.is_file(), f"missing runtime contract library: {CONTRACT_LIB}"
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; shift; "$@"',
            "bash",
            str(CONTRACT_LIB),
            function,
            *args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_quota_evidence_requires_auditable_b300_capacity(tmp_path: Path) -> None:
    evidence = tmp_path / "quota.env"
    evidence.write_text(
        "\n".join(
            (
                "B300_QUOTA_GPU_TYPE=B300",
                "B300_QUOTA_CHARGED_GROUP=b300_train_infra",
                "B300_QUOTA_AVAILABLE_GPUS=16",
                "B300_QUOTA_OBSERVED_AT=2026-08-17T19:30:00+08:00",
                "B300_QUOTA_SOURCE=quota-owner",
            )
        )
        + "\n"
    )

    result = _run_contract(
        "require_b300_quota_evidence",
        str(evidence),
        "16",
        "b300_train_infra",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "16"


def test_quota_evidence_rejects_insufficient_or_wrong_capacity(
    tmp_path: Path,
) -> None:
    cases = {
        "insufficient": (
            "B300_QUOTA_GPU_TYPE=B300\n"
            "B300_QUOTA_CHARGED_GROUP=b300_train_infra\n"
            "B300_QUOTA_AVAILABLE_GPUS=15\n"
            "B300_QUOTA_OBSERVED_AT=2026-08-17T19:30:00+08:00\n"
            "B300_QUOTA_SOURCE=quota-owner\n"
        ),
        "wrong_gpu": (
            "B300_QUOTA_GPU_TYPE=H800\n"
            "B300_QUOTA_CHARGED_GROUP=b300_train_infra\n"
            "B300_QUOTA_AVAILABLE_GPUS=16\n"
            "B300_QUOTA_OBSERVED_AT=2026-08-17T19:30:00+08:00\n"
            "B300_QUOTA_SOURCE=quota-owner\n"
        ),
        "wrong_group": (
            "B300_QUOTA_GPU_TYPE=B300\n"
            "B300_QUOTA_CHARGED_GROUP=codesign\n"
            "B300_QUOTA_AVAILABLE_GPUS=16\n"
            "B300_QUOTA_OBSERVED_AT=2026-08-17T19:30:00+08:00\n"
            "B300_QUOTA_SOURCE=quota-owner\n"
        ),
        "missing_source": (
            "B300_QUOTA_GPU_TYPE=B300\n"
            "B300_QUOTA_CHARGED_GROUP=b300_train_infra\n"
            "B300_QUOTA_AVAILABLE_GPUS=16\n"
            "B300_QUOTA_OBSERVED_AT=2026-08-17T19:30:00+08:00\n"
        ),
    }

    for name, content in cases.items():
        evidence = tmp_path / f"{name}.env"
        evidence.write_text(content)
        result = _run_contract(
            "require_b300_quota_evidence",
            str(evidence),
            "16",
            "b300_train_infra",
        )
        assert result.returncode != 0, name


def test_cleanup_inventory_requires_successful_queries(tmp_path: Path) -> None:
    rjob = tmp_path / "rjob.log"
    replicas = tmp_path / "replicas.log"
    rjob.write_text("")
    replicas.write_text("NAME READY STATUS\n")

    clean = _run_contract(
        "cleanup_inventory_is_empty",
        "0",
        "0",
        str(rjob),
        str(replicas),
        "s4p-test",
    )
    failed_rjob_query = _run_contract(
        "cleanup_inventory_is_empty",
        "1",
        "0",
        str(rjob),
        str(replicas),
        "s4p-test",
    )
    failed_replica_query = _run_contract(
        "cleanup_inventory_is_empty",
        "0",
        "1",
        str(rjob),
        str(replicas),
        "s4p-test",
    )

    assert clean.returncode == 0, clean.stderr
    assert failed_rjob_query.returncode != 0
    assert failed_replica_query.returncode != 0


def test_cleanup_inventory_rejects_remaining_resources(tmp_path: Path) -> None:
    rjob = tmp_path / "rjob.log"
    replicas = tmp_path / "replicas.log"
    rjob.write_text("s4p-test Running\n")
    replicas.write_text("NAME READY STATUS\n")

    result = _run_contract(
        "cleanup_inventory_is_empty",
        "0",
        "0",
        str(rjob),
        str(replicas),
        "s4p-test",
    )

    assert result.returncode != 0


def test_runtime_log_gate_rejects_failure_markers(tmp_path: Path) -> None:
    clean_log = tmp_path / "clean.log"
    clean_log.write_text("Using AgRsAll2AllManager all2all manager\n")
    assert _run_contract("assert_runtime_log_clean", str(clean_log)).returncode == 0

    for index, marker in enumerate(("Traceback", "ERROR", "Broken pipe")):
        failure_log = tmp_path / f"failure_{index}.log"
        failure_log.write_text(f"runtime marker: {marker}\n")
        assert _run_contract("assert_runtime_log_clean", str(failure_log)).returncode != 0
