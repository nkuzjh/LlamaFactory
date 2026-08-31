#!/usr/bin/env bash
set -Eeuo pipefail

# Fail-closed serial handoff:
#   Text250k exp4_4_2/exp4_7_2 complete
#   -> both Text250k alignment evaluations are already complete
#   -> PT-exp2-mm-rowbal-cont3 train
#   -> ep1 predict/evaluate -> ep2 predict/evaluate -> ep3 predict/evaluate
#
# This wrapper owns CUDA 1 only after the upstream wrapper has exited.  It
# never sends a signal to an existing process and never attempts to recover a
# partial run implicitly.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON="/home/jiahao/miniconda3/envs/llamafactory/bin/python"
UPSTREAM_WRAPPER="${SCRIPT_DIR}/run_exp4_4_2_exp4_7_2_pt_exp2_text250k_cuda1.sh"
UPSTREAM_PID_FILE="${SCRIPT_DIR}/run_exp4_4_2_exp4_7_2_pt_exp2_text250k_cuda1.pid"
UPSTREAM_LOG_FILE="${SCRIPT_DIR}/run_exp4_4_2_exp4_7_2_pt_exp2_text250k_cuda1.log"
DOWNSTREAM_LAUNCHER="${ROOT}/scripts/launch_bricknet_pt_exp2_text250k_downstream.py"
ROWBAL_LAUNCHER="${ROOT}/scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3.py"

LOCK_FILE="${SCRIPT_DIR}/run_pt_exp2_mm_rowbal_cont3_after_text250k_cuda1.lock"
PID_FILE="${SCRIPT_DIR}/run_pt_exp2_mm_rowbal_cont3_after_text250k_cuda1.pid"
LOG_FILE="${SCRIPT_DIR}/run_pt_exp2_mm_rowbal_cont3_after_text250k_cuda1.log"
MIN_FREE_KIB=$((40 * 1024 * 1024))
POLL_SECONDS=30

UPSTREAM_MODE=""
UPSTREAM_PID=""
UPSTREAM_START_TICKS=""
UPSTREAM_LOG_OFFSET=""
PID_OWNED=0
TEMP_FILES=()
NEW_TEMP_FILE=""

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" >&2
}

on_error() {
    local exit_code="${1:-1}"
    local line_no="${2:-unknown}"
    log "ERROR: exit=${exit_code} line=${line_no} command=${BASH_COMMAND:-unknown}"
}

cleanup_temps() {
    local path
    for path in "${TEMP_FILES[@]}"; do
        [[ -z "${path}" ]] || rm -f -- "${path}" || true
    done
}

on_exit() {
    local exit_code="$?"
    set +e
    cleanup_temps
    if ((PID_OWNED)); then
        local current_pid=""
        if [[ -f "${PID_FILE}" ]]; then
            current_pid="$(tr -d '[:space:]' <"${PID_FILE}" 2>/dev/null || true)"
        fi
        if [[ "${current_pid}" == "$$" ]]; then
            rm -f -- "${PID_FILE}" || true
        fi
    fi
    log "handoff exit status=${exit_code}"
}

trap 'on_error "$?" "$LINENO"' ERR
trap on_exit EXIT

if (($# != 0)); then
    printf 'usage: %s\n' "$0" >&2
    exit 2
fi

mkdir -p -- "${SCRIPT_DIR}"
touch -- "${LOG_FILE}"
exec >>"${LOG_FILE}" 2>&1

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    log "handoff start refused: another handoff wrapper holds the lock"
    exit 1
fi

printf '%s\n' "$$" >"${PID_FILE}"
PID_OWNED=1
log "handoff start pid=$$"

die() {
    log "ERROR: $*"
    exit 1
}

read_pid_value() {
    local path="$1"
    local raw
    local value
    [[ -f "${path}" ]] || return 1
    if ! raw="$(<"${path}")"; then
        return 1
    fi
    [[ "${raw}" =~ ^[[:space:]]*[1-9][0-9]*[[:space:]]*$ ]] || return 1
    value="${raw//[[:space:]]/}"
    printf '%s' "${value}"
}

pid_start_ticks() {
    local pid="$1"
    [[ -r "/proc/${pid}/stat" ]] || return 1
    awk '{print $22}' "/proc/${pid}/stat"
}

upstream_process_status() {
    # Return 0 for the same live process, 1 for an exited/missing process,
    # and 2 for PID reuse or command-line identity drift.
    local pid="$1"
    local expected_start="$2"
    local current_start
    local state
    local cmdline

    [[ -e "/proc/${pid}/stat" ]] || return 1
    [[ -r "/proc/${pid}/stat" ]] || return 2
    if ! current_start="$(pid_start_ticks "${pid}")"; then
        return 2
    fi
    if ! state="$(awk '{print $3}' "/proc/${pid}/stat")"; then
        return 2
    fi
    [[ "${state}" == "Z" ]] && return 1
    [[ "${current_start}" == "${expected_start}" ]] || return 2
    [[ -r "/proc/${pid}/cmdline" ]] || return 2
    if ! cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"; then
        return 2
    fi
    [[ "${cmdline}" == *"${UPSTREAM_WRAPPER}"* ]] || return 2
    return 0
}

check_upstream_pidfile_same() {
    local current_pid
    if ! current_pid="$(read_pid_value "${UPSTREAM_PID_FILE}")"; then
        return 1
    fi
    [[ "${current_pid}" == "${UPSTREAM_PID}" ]]
}

pin_upstream() {
    if [[ ! -e "${UPSTREAM_PID_FILE}" ]]; then
        UPSTREAM_MODE="fallback"
        UPSTREAM_LOG_OFFSET=""
        log "upstream pidfile absent at handoff start; old upstream log will not be trusted"
        return
    fi

    UPSTREAM_MODE="tracked"
    [[ -f "${UPSTREAM_PID_FILE}" ]] || die "upstream pidfile is not a regular file"
    UPSTREAM_PID="$(read_pid_value "${UPSTREAM_PID_FILE}")" \
        || die "upstream pidfile is invalid"
    [[ -f "${UPSTREAM_WRAPPER}" ]] || die "upstream wrapper is missing"
    [[ -f "${UPSTREAM_LOG_FILE}" ]] || die "upstream log is missing"

    UPSTREAM_START_TICKS="$(pid_start_ticks "${UPSTREAM_PID}")" \
        || die "upstream PID is not live"
    upstream_process_status "${UPSTREAM_PID}" "${UPSTREAM_START_TICKS}" \
        || die "upstream PID is not the expected wrapper process"
    UPSTREAM_LOG_OFFSET="$(stat -c '%s' -- "${UPSTREAM_LOG_FILE}")" \
        || die "cannot record upstream log byte offset"
    check_upstream_pidfile_same \
        || die "upstream pidfile changed while handoff was pinning it"
    log "upstream pinned pid=${UPSTREAM_PID} log_offset_bytes=${UPSTREAM_LOG_OFFSET}"
}

wait_for_tracked_upstream() {
    local status
    local pidfile_present
    while :; do
        if [[ -e "${UPSTREAM_PID_FILE}" ]]; then
            check_upstream_pidfile_same \
                || die "upstream pidfile changed before the pinned PID exited"
        fi

        if upstream_process_status "${UPSTREAM_PID}" "${UPSTREAM_START_TICKS}"; then
            if [[ -e "${UPSTREAM_PID_FILE}" ]]; then
                pidfile_present=true
            else
                pidfile_present=false
            fi
            log "upstream wait pid=${UPSTREAM_PID} still_running=true pidfile_present=${pidfile_present}"
            sleep "${POLL_SECONDS}"
            continue
        else
            status="$?"
        fi
        if ((status == 2)); then
            die "upstream PID identity changed while waiting"
        fi
        # The upstream EXIT trap removes its PID file before writing the
        # successful exit marker.  Conversely, a just-dead shell can briefly
        # leave the PID file visible.  In either case the pinned PID/start
        # ticks remain the authority; the suffix gate below proves completion.
        if [[ -e "${UPSTREAM_PID_FILE}" ]]; then
            check_upstream_pidfile_same \
                || die "upstream pidfile changed while the pinned PID was exiting"
        fi
        log "upstream exact PID exited; allowing pidfile cleanup window"
        break
    done
    log "upstream process exited; validating only new log bytes"
}

validate_upstream_log_suffix() {
    local reason
    if ! reason="$("${PYTHON}" - "${UPSTREAM_LOG_FILE}" "${UPSTREAM_LOG_OFFSET}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
offset = int(sys.argv[2])
try:
    size = path.stat().st_size
    if size < offset:
        print("UPSTREAM_LOG_TRUNCATED")
        raise SystemExit(1)
    payload = path.read_bytes()[offset:]
except (OSError, ValueError):
    print("UPSTREAM_LOG_UNREADABLE")
    raise SystemExit(1)

if b"ERROR:" in payload:
    print("UPSTREAM_LOG_CONTAINS_ERROR")
    raise SystemExit(1)
if b"Text250k-only downstream batch complete" not in payload:
    print("UPSTREAM_BATCH_COMPLETE_MARKER_MISSING")
    raise SystemExit(1)
status_lines = [
    line.strip()
    for line in payload.splitlines()
    if b"Text250k-only downstream batch exit status=" in line
]
if not any(line.endswith(b"=0") for line in status_lines):
    print("UPSTREAM_EXIT_STATUS_ZERO_MARKER_MISSING")
    raise SystemExit(1)
for line in status_lines:
    if not line.endswith(b"=0"):
        print("UPSTREAM_NONZERO_EXIT_STATUS_MARKER")
        raise SystemExit(1)
for marker in (
    b"Traceback (most recent call last):",
    b"Text250k-only downstream launch blocked;",
):
    if marker in payload:
        print("UPSTREAM_FATAL_MARKER_PRESENT")
        raise SystemExit(1)
print("ok")
PY
)"; then
        die "upstream log validation failed: ${reason:-unknown}"
    fi
    [[ "${reason}" == "ok" ]] || die "upstream log validation returned an invalid result"
    log "upstream completion evidence validated offset=${UPSTREAM_LOG_OFFSET}"
}

check_disk() {
    local available_kib
    available_kib="$(df -Pk -- "${ROOT}" | awk 'NR == 2 {print $4}')" \
        || die "disk gate could not query filesystem"
    [[ "${available_kib}" =~ ^[0-9]+$ ]] \
        || die "disk gate returned an invalid free-space value"
    ((available_kib >= MIN_FREE_KIB)) \
        || die "disk gate failed: less than 40 GiB is available"
    log "disk gate passed free_kib=${available_kib}"
}

check_gpu1_idle() {
    local gpu1_uuid
    local processes
    local process_count
    command -v nvidia-smi >/dev/null 2>&1 \
        || die "CUDA1 gate failed: nvidia-smi is unavailable"
    gpu1_uuid="$(nvidia-smi --id=1 --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" \
        || die "CUDA1 gate failed: physical GPU 1 UUID is unavailable"
    [[ -n "${gpu1_uuid}" ]] \
        || die "CUDA1 gate failed: physical GPU 1 UUID is empty"
    processes="$(nvidia-smi --id="${gpu1_uuid}" \
        --query-compute-apps=pid,used_memory,process_name \
        --format=csv,noheader,nounits 2>/dev/null)" \
        || die "CUDA1 gate failed: compute process query failed"
    process_count="$(printf '%s\n' "${processes}" | awk 'NF {count += 1} END {print count + 0}')"
    ((process_count == 0)) \
        || die "CUDA1 gate failed: compute processes are present"
    log "CUDA1 gate passed compute_processes=0"
}

assert_handoff_window_open() {
    # A new upstream wrapper must never start between stages.  In fallback
    # mode the PID file must remain absent.  In tracked mode the original
    # PID file may still be in the upstream EXIT-trap cleanup window; an exact
    # stale PID is allowed only when its pinned process identity is gone.
    if [[ "${UPSTREAM_MODE}" == "fallback" ]]; then
        [[ ! -e "${UPSTREAM_PID_FILE}" ]] \
            || die "handoff window changed: upstream pidfile reappeared"
        return
    fi
    if [[ ! -e "${UPSTREAM_PID_FILE}" ]]; then
        return
    fi
    check_upstream_pidfile_same \
        || die "handoff window changed: upstream pidfile contains another PID"
    if upstream_process_status "${UPSTREAM_PID}" "${UPSTREAM_START_TICKS}"; then
        die "handoff window changed: pinned upstream PID is still running"
    else
        local status="$?"
    fi
    ((status == 1)) \
        || die "handoff window changed: upstream PID identity was reused"
}

check_resources_before_launcher() {
    assert_handoff_window_open
    check_disk
    check_gpu1_idle
}

new_temp_file() {
    NEW_TEMP_FILE="$(mktemp "${SCRIPT_DIR}/.rowbal-handoff.XXXXXX")" \
        || die "cannot create temporary gate file"
    TEMP_FILES+=("${NEW_TEMP_FILE}")
}

run_text250k_evaluate_gate() {
    local experiment="$1"
    local result_file
    local stderr_file
    local reason

    log "stage_start name=text250k_${experiment}_evaluate_dry_run"
    check_resources_before_launcher
    new_temp_file
    result_file="${NEW_TEMP_FILE}"
    new_temp_file
    stderr_file="${NEW_TEMP_FILE}"
    if ! "${PYTHON}" "${DOWNSTREAM_LAUNCHER}" \
        --run "${experiment}" \
        --action evaluate \
        --gpus 1 \
        >"${result_file}" 2>"${stderr_file}"; then
        log "stage_failed name=text250k_${experiment}_evaluate_dry_run reason=launcher_exit"
        return 1
    fi
    if ! reason="$("${PYTHON}" - "${result_file}" "${experiment}" <<'PY'
import json
import sys

path, expected_run = sys.argv[1:]
try:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
except (OSError, UnicodeDecodeError, json.JSONDecodeError):
    print("INVALID_JSON")
    raise SystemExit(1)

checks = payload.get("checks") if isinstance(payload, dict) else None
strict = (
    isinstance(payload, dict)
    and payload.get("experiment") == expected_run
    and payload.get("action") == "evaluate"
    and payload.get("gpus") == "1"
    and payload.get("mode") == "dry-run"
    and payload.get("executed") is False
    and payload.get("ready") is True
    and payload.get("already_complete") is True
    and isinstance(checks, dict)
    and checks.get("run") == expected_run
    and checks.get("output_complete") is True
    and payload.get("blockers") == []
)
if not strict:
    print("EVALUATE_GATE_NOT_STRICTLY_COMPLETE")
    raise SystemExit(1)
print("ok")
PY
)"; then
        log "stage_failed name=text250k_${experiment}_evaluate_dry_run reason=${reason:-invalid_gate}"
        return 1
    fi
    [[ "${reason}" == "ok" ]] \
        || die "Text250k evaluate gate returned an invalid parser result"
    log "stage_complete name=text250k_${experiment}_evaluate_dry_run"
}

run_rowbal_stage() {
    local action="$1"
    local run_name="${2:-}"
    local stage_name="rowbal_${action}"
    local command=("${PYTHON}" "${ROWBAL_LAUNCHER}" "--action" "${action}")

    if [[ -n "${run_name}" ]]; then
        stage_name+="_${run_name}"
        command+=(--run "${run_name}")
    fi
    command+=(--gpus 1 --execute)

    log "stage_start name=${stage_name}"
    check_resources_before_launcher
    if ! "${command[@]}"; then
        log "stage_failed name=${stage_name} reason=launcher_exit"
        return 1
    fi
    log "stage_complete name=${stage_name}"
}

pin_upstream
if [[ "${UPSTREAM_MODE}" == "tracked" ]]; then
    wait_for_tracked_upstream
    validate_upstream_log_suffix
else
    log "handoff fallback requires CUDA1 idle and both strict evaluate dry-runs"
fi

run_text250k_evaluate_gate exp4_4_2
run_text250k_evaluate_gate exp4_7_2
log "handoff validated mode=${UPSTREAM_MODE}"

run_rowbal_stage train
run_rowbal_stage predict ep1
run_rowbal_stage evaluate ep1
run_rowbal_stage predict ep2
run_rowbal_stage evaluate ep2
run_rowbal_stage predict ep3
run_rowbal_stage evaluate ep3

log "overall complete experiment=PT-exp2-mm-rowbal-cont3"
