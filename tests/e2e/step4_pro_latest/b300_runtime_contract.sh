#!/usr/bin/env bash

_contract_error() {
    printf '%s\n' "$*" >&2
    return 1
}

_required_evidence_field() {
    local evidence_file="${1:?evidence_file is required}"
    local field_name="${2:?field_name is required}"
    local field_count
    local field_value

    field_count="$(
        {
            grep -Ec "^${field_name}=" "${evidence_file}" || true
        }
    )"
    if [[ "${field_count}" != "1" ]]; then
        _contract_error \
            "Expected exactly one ${field_name} field in ${evidence_file}"
        return 1
    fi
    field_value="$(
        sed -n "s/^${field_name}=//p" "${evidence_file}"
    )"
    if [[ -z "${field_value}" ]]; then
        _contract_error \
            "Evidence field ${field_name} must not be empty: ${evidence_file}"
        return 1
    fi
    printf '%s\n' "${field_value}"
}

require_b300_quota_evidence() {
    local evidence_file="${1:?evidence_file is required}"
    local required_gpus="${2:?required_gpus is required}"
    local charged_group="${3:?charged_group is required}"
    local gpu_type
    local evidence_group
    local available_gpus

    if [[ ! -s "${evidence_file}" ]]; then
        _contract_error "B300 quota evidence is missing or empty: ${evidence_file}"
        return 1
    fi
    case "${evidence_file}" in
        /tmp | /tmp/*)
            _contract_error \
                "B300 quota evidence must use disk-backed storage: ${evidence_file}"
            return 1
            ;;
    esac
    if [[ ! "${required_gpus}" =~ ^[0-9]+$ ]] || (( required_gpus <= 0 )); then
        _contract_error "required_gpus must be a positive integer: ${required_gpus}"
        return 1
    fi

    gpu_type="$(
        _required_evidence_field "${evidence_file}" B300_QUOTA_GPU_TYPE
    )" || return 1
    evidence_group="$(
        _required_evidence_field \
            "${evidence_file}" B300_QUOTA_CHARGED_GROUP
    )" || return 1
    available_gpus="$(
        _required_evidence_field \
            "${evidence_file}" B300_QUOTA_AVAILABLE_GPUS
    )" || return 1
    _required_evidence_field \
        "${evidence_file}" B300_QUOTA_OBSERVED_AT >/dev/null || return 1
    _required_evidence_field \
        "${evidence_file}" B300_QUOTA_SOURCE >/dev/null || return 1

    if [[ "${gpu_type}" != "B300" ]]; then
        _contract_error "Quota evidence GPU type must be B300: ${gpu_type}"
        return 1
    fi
    if [[ "${evidence_group}" != "${charged_group}" ]]; then
        _contract_error \
            "Quota evidence charged group mismatch: expected=${charged_group} actual=${evidence_group}"
        return 1
    fi
    if [[ ! "${available_gpus}" =~ ^[0-9]+$ ]]; then
        _contract_error \
            "B300_QUOTA_AVAILABLE_GPUS must be an integer: ${available_gpus}"
        return 1
    fi
    if (( available_gpus < required_gpus )); then
        _contract_error \
            "Insufficient B300 quota evidence: required=${required_gpus} available=${available_gpus}"
        return 1
    fi

    printf '%s\n' "${available_gpus}"
}

cleanup_inventory_is_empty() {
    local rjob_query_status="${1:?rjob_query_status is required}"
    local replica_query_status="${2:?replica_query_status is required}"
    local rjob_inventory="${3:?rjob_inventory is required}"
    local replica_inventory="${4:?replica_inventory is required}"
    local rjob_name="${5:?rjob_name is required}"

    if [[ "${rjob_query_status}" != "0" ||
        "${replica_query_status}" != "0" ]]; then
        _contract_error \
            "Cleanup inventory query failed: rjob_status=${rjob_query_status} replica_status=${replica_query_status}"
        return 1
    fi
    if [[ ! -f "${rjob_inventory}" || ! -f "${replica_inventory}" ]]; then
        _contract_error "Cleanup inventory output is missing"
        return 1
    fi
    if grep -Fq "${rjob_name}" "${rjob_inventory}" "${replica_inventory}"; then
        _contract_error "Cleanup inventory still contains ${rjob_name}"
        return 1
    fi
}

assert_runtime_log_clean() {
    local runtime_log="${1:?runtime_log is required}"
    local failure_pattern='Traceback|Broken pipe|(^|[[:space:]])ERROR([[:space:]:]|$)'

    if [[ ! -f "${runtime_log}" ]]; then
        _contract_error "Runtime log is missing: ${runtime_log}"
        return 1
    fi
    if grep -Eq "${failure_pattern}" "${runtime_log}"; then
        printf 'Runtime failure marker found in %s:\n' "${runtime_log}" >&2
        grep -En "${failure_pattern}" "${runtime_log}" >&2 || true
        return 1
    fi
}
