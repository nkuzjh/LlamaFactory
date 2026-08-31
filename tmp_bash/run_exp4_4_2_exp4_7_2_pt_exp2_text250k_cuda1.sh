#!/usr/bin/env bash
set -Eeuo pipefail

# Text250k-only downstream is a fixed physical-CUDA-1 serial chain.  Each
# stage is independently gated by the launcher; this wrapper never sends a
# signal to an existing process and stops on the first failed stage.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON="/home/jiahao/miniconda3/envs/llamafactory/bin/python"
LAUNCHER="${ROOT}/scripts/launch_bricknet_pt_exp2_text250k_downstream.py"
LOCK_FILE="${SCRIPT_DIR}/run_exp4_4_2_exp4_7_2_pt_exp2_text250k_cuda1.lock"
LOG_FILE="${SCRIPT_DIR}/run_exp4_4_2_exp4_7_2_pt_exp2_text250k_cuda1.log"
PID_FILE="${SCRIPT_DIR}/run_exp4_4_2_exp4_7_2_pt_exp2_text250k_cuda1.pid"
MIN_FREE_KIB=$((40 * 1024 * 1024))

if [[ "${1:-}" == "--preflight-only" ]]; then
    PREFLIGHT_ONLY=1
    shift
else
    PREFLIGHT_ONLY=0
fi
if (($# != 0)); then
    echo "usage: $0 [--preflight-only]" >&2
    exit 2
fi

mkdir -p -- "${SCRIPT_DIR}"
touch -- "${LOG_FILE}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "another Text250k downstream batch already holds ${LOCK_FILE}" >&2
    exit 1
fi
printf '%s\n' "$$" >"${PID_FILE}"
exec >>"${LOG_FILE}" 2>&1

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

on_error() {
    local exit_code="$1"
    local line_no="$2"
    local command_text="$3"
    log "ERROR: exit=${exit_code} line=${line_no} command=${command_text}"
}

on_exit() {
    local exit_code="$?"
    rm -f -- "${PID_FILE}"
    log "Text250k-only downstream batch exit status=${exit_code}"
}

trap 'on_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
trap on_exit EXIT

log "Text250k-only downstream batch start preflight_only=${PREFLIGHT_ONLY} pid=$$"

check_disk() {
    local available_kib
    available_kib="$(df -Pk "${ROOT}" | awk 'NR == 2 {print $4}')"
    if [[ ! "${available_kib}" =~ ^[0-9]+$ ]] || ((available_kib < MIN_FREE_KIB)); then
        echo "disk gate failed: ${available_kib:-unknown} KiB free; need ${MIN_FREE_KIB} KiB" >&2
        return 1
    fi
    log "disk gate passed: ${available_kib} KiB free"
}

check_gpu1() {
    local gpu1_uuid
    local processes
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "GPU1 occupancy gate failed: nvidia-smi is unavailable" >&2
        return 1
    fi
    if ! gpu1_uuid="$(nvidia-smi --id=1 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')" || [[ -z "${gpu1_uuid}" ]]; then
        echo "GPU1 occupancy gate failed: could not resolve physical GPU 1 UUID" >&2
        return 1
    fi
    log "physical GPU 1 UUID: ${gpu1_uuid}"
    if ! processes="$(nvidia-smi --id="${gpu1_uuid}" --query-compute-apps=pid,used_memory,process_name --format=csv,noheader,nounits)"; then
        echo "GPU1 occupancy gate failed: nvidia-smi could not query physical GPU 1" >&2
        return 1
    fi
    if [[ -n "${processes}" ]]; then
        echo "GPU1 occupancy gate failed; refusing to alter existing processes:" >&2
        echo "${processes}" >&2
        return 1
    fi
    log "GPU1 occupancy gate passed: no compute process"
}

run_stage() {
    local experiment="$1"
    local action="$2"
    check_disk
    check_gpu1
    local command=(
        "${PYTHON}"
        "${LAUNCHER}"
        --run "${experiment}"
        --action "${action}"
        --gpus 1
        --text250k-approved
    )
    if ((PREFLIGHT_ONLY == 0)); then
        command+=(--execute)
    fi
    log "+ ${command[*]}"
    if ((PREFLIGHT_ONLY == 0)); then
        "${command[@]}"
        return
    fi

    local result_file
    result_file="$(mktemp)"
    if ! "${command[@]}" >"${result_file}"; then
        cat -- "${result_file}"
        rm -f -- "${result_file}"
        return 1
    fi
    cat -- "${result_file}"
    if ! "${PYTHON}" - "${result_file}" "${action}" <<'PY'
import json
import sys

path, action = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
blockers = payload.get("blockers")
allowed = {
    "train": set(),
    "predict": {"WAIT_FINAL_TRAIN_ADAPTER"},
    "evaluate": {"WAIT_FINAL_TRAIN_ADAPTER", "WAIT_FINAL_PREDICTION"},
}[action]
actual = set(blockers if isinstance(blockers, list) else ())
if not actual.issubset(allowed):
    print(
        f"preflight blocker set for {action} is {sorted(actual)}, "
        f"allowed={sorted(allowed)}",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"preflight accepted for {action}: blockers={sorted(actual)}")
PY
    then
        rm -f -- "${result_file}"
        return 1
    fi
    rm -f -- "${result_file}"
}

# Fail-stop ordering is intentional: each completed/no-op stage is checked
# before the following stage consumes its artifacts.
run_stage exp4_4_2 train
run_stage exp4_4_2 predict
run_stage exp4_4_2 evaluate
run_stage exp4_7_2 train
run_stage exp4_7_2 predict
run_stage exp4_7_2 evaluate

log "Text250k-only downstream batch complete"
