#!/usr/bin/env bash
set -Eeuo pipefail

# Wait for physical CUDA0 without touching its current owner, then hand off to
# the already validated fail-stop serial wrapper.  Launch this waiter with
# nohup + setsid so the experiment does not depend on an SSH/Codex session.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
WRAPPER="${SCRIPT_DIR}/run_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.sh"
WAIT_LOCK="${SCRIPT_DIR}/wait_for_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.lock"
WAIT_PID="${SCRIPT_DIR}/wait_for_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.pid"
WAIT_LOG="${SCRIPT_DIR}/wait_for_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.log"
WRAPPER_PID="${SCRIPT_DIR}/run_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.pid"
POLL_SECONDS=60
MIN_FREE_KIB=$((40 * 1024 * 1024))
PID_OWNED=0

if (($# != 0)); then
    printf 'usage: %s\n' "$0" >&2
    exit 2
fi

mkdir -p -- "${SCRIPT_DIR}"
touch -- "${WAIT_LOG}"
exec >>"${WAIT_LOG}" 2>&1
exec 9>"${WAIT_LOCK}"
if ! flock -n 9; then
    printf '[%s] another CUDA0 waiter already holds %s\n' \
        "$(date --iso-8601=seconds)" "${WAIT_LOCK}"
    exit 1
fi

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

cleanup() {
    local exit_code="$?"
    set +e
    if ((PID_OWNED)) && [[ -f "${WAIT_PID}" ]]; then
        local recorded_pid
        recorded_pid="$(tr -d '[:space:]' <"${WAIT_PID}" 2>/dev/null || true)"
        if [[ "${recorded_pid}" == "$$" ]]; then
            rm -f -- "${WAIT_PID}"
        fi
    fi
    log "CUDA0 waiter exit status=${exit_code}"
}

handle_signal() {
    local signal_name="$1"
    log "received ${signal_name}; exiting without touching any GPU process"
    if [[ "${signal_name}" == "INT" ]]; then
        exit 130
    fi
    exit 143
}

trap cleanup EXIT
trap 'handle_signal TERM' TERM
trap 'handle_signal INT' INT

printf '%s\n' "$$" >"${WAIT_PID}"
PID_OWNED=1
log "CUDA0 waiter start pid=$$ poll_seconds=${POLL_SECONDS}"

gpu0_idle() {
    local gpu_uuid
    local apps
    if ! gpu_uuid="$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" || [[ -z "${gpu_uuid}" ]]; then
        log "cannot resolve physical CUDA0 UUID; remaining queued"
        return 1
    fi
    if ! apps="$(nvidia-smi --id="${gpu_uuid}" \
        --query-compute-apps=pid,used_memory,process_name \
        --format=csv,noheader,nounits 2>/dev/null)"; then
        log "cannot query physical CUDA0 compute apps; remaining queued"
        return 1
    fi
    if [[ -z "${apps//[[:space:]]/}" ]]; then
        log "physical CUDA0 is idle uuid=${gpu_uuid}"
        return 0
    fi

    log "physical CUDA0 is busy; no process will be killed or interrupted"
    while IFS= read -r row; do
        [[ -z "${row//[[:space:]]/}" ]] && continue
        local gpu_pid="${row%%,*}"
        gpu_pid="${gpu_pid//[[:space:]]/}"
        local identity="unavailable"
        if [[ "${gpu_pid}" =~ ^[1-9][0-9]*$ ]]; then
            identity="$(ps -ww -o user=,pid=,lstart=,args= -p "${gpu_pid}" 2>/dev/null || true)"
            [[ -n "${identity}" ]] || identity="unavailable"
        fi
        log "CUDA0 occupant nvidia_row=${row} identity=${identity}"
    done <<<"${apps}"
    return 1
}

disk_ready() {
    local free_kib
    free_kib="$(df -Pk -- "${ROOT}" | awk 'NR == 2 {print $4}')"
    if [[ ! "${free_kib}" =~ ^[0-9]+$ ]] || ((free_kib < MIN_FREE_KIB)); then
        log "disk gate not ready free_kib=${free_kib:-unknown} minimum_free_kib=${MIN_FREE_KIB}"
        return 1
    fi
    log "disk gate passed free_kib=${free_kib} minimum_free_kib=${MIN_FREE_KIB}"
}

wrapper_active() {
    local pid
    [[ -f "${WRAPPER_PID}" ]] || return 1
    pid="$(tr -d '[:space:]' <"${WRAPPER_PID}" 2>/dev/null || true)"
    if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; then
        log "serial wrapper already active pid=${pid}; remaining queued"
        return 0
    fi
    log "serial wrapper PID file is stale or invalid; wrapper lock remains authoritative"
    return 1
}

while :; do
    if wrapper_active; then
        sleep "${POLL_SECONDS}"
        continue
    fi
    if ! gpu0_idle; then
        sleep "${POLL_SECONDS}"
        continue
    fi
    if ! disk_ready; then
        sleep "${POLL_SECONDS}"
        continue
    fi

    log "CUDA0 and disk gates passed; running the existing wrapper preflight"
    if ! "${WRAPPER}" --preflight-only; then
        log "wrapper preflight did not pass; remaining queued"
        sleep "${POLL_SECONDS}"
        continue
    fi

    # Close the race window created by the preflight.  The wrapper repeats
    # these checks before every train/predict/evaluate stage.
    if wrapper_active || ! gpu0_idle || ! disk_ready; then
        log "handoff gates changed during preflight; remaining queued"
        sleep "${POLL_SECONDS}"
        continue
    fi

    log "handoff gates passed; execing the complete serial wrapper"
    rm -f -- "${WAIT_PID}"
    PID_OWNED=0
    exec "${WRAPPER}"
done
