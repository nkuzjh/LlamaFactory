#!/usr/bin/env bash
set -Eeuo pipefail

# PT-exp2-mm-rowbal-cont3 ep3 downstream is a fixed physical-CUDA-0 serial
# chain.  The launcher owns the artifact/data/config gates and safely no-ops
# only complete stages.  This wrapper refuses to alter an existing GPU
# process and stops at the first failed stage.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON="/home/jiahao/miniconda3/envs/llamafactory/bin/python"
LAUNCHER="${ROOT}/scripts/launch_bricknet_pt_exp2_mm_rowbal_cont3_downstream.py"
LOCK_FILE="${SCRIPT_DIR}/run_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.lock"
LOG_FILE="${SCRIPT_DIR}/run_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.log"
PID_FILE="${SCRIPT_DIR}/run_exp4_4_3_exp4_7_3_pt_exp2_mm_rowbal_cont3_cuda0.pid"
MIN_FREE_KIB=$((40 * 1024 * 1024))
PID_OWNED=0

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
    echo "another PT-exp2-mm-rowbal-cont3 downstream batch already holds ${LOCK_FILE}" >&2
    exit 1
fi

printf '%s\n' "$$" >"${PID_FILE}"
PID_OWNED=1
exec >>"${LOG_FILE}" 2>&1

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

on_error() {
    local exit_code="${1:-1}"
    local line_no="${2:-unknown}"
    log "ERROR: exit=${exit_code} line=${line_no} command=${BASH_COMMAND:-unknown}"
}

on_exit() {
    local exit_code="$?"
    set +e
    if ((PID_OWNED)); then
        local current_pid=""
        if [[ -f "${PID_FILE}" ]]; then
            current_pid="$(tr -d '[:space:]' <"${PID_FILE}" 2>/dev/null || true)"
        fi
        if [[ "${current_pid}" == "$$" ]]; then
            rm -f -- "${PID_FILE}" || true
        fi
    fi
    log "PT-exp2-mm-rowbal-cont3 downstream CUDA0 batch exit status=${exit_code}"
}

trap 'on_error "$?" "$LINENO"' ERR
trap on_exit EXIT

log "PT-exp2-mm-rowbal-cont3 downstream CUDA0 batch start preflight_only=${PREFLIGHT_ONLY} pid=$$"

check_disk() {
    local available_kib
    available_kib="$(df -Pk "${ROOT}" | awk 'NR == 2 {print $4}')"
    if [[ ! "${available_kib}" =~ ^[0-9]+$ ]] || ((available_kib < MIN_FREE_KIB)); then
        echo "disk gate failed: ${available_kib:-unknown} KiB free; need ${MIN_FREE_KIB} KiB" >&2
        return 1
    fi
    log "disk gate passed: ${available_kib} KiB free"
}

check_gpu0() {
    local gpu0_uuid
    local processes
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "GPU0 occupancy gate failed: nvidia-smi is unavailable" >&2
        return 1
    fi
    if ! gpu0_uuid="$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')" || [[ -z "${gpu0_uuid}" ]]; then
        echo "GPU0 occupancy gate failed: could not resolve physical GPU 0 UUID" >&2
        return 1
    fi
    log "physical GPU 0 UUID: ${gpu0_uuid}"
    if ! processes="$(nvidia-smi --id="${gpu0_uuid}" --query-compute-apps=pid,used_memory,process_name --format=csv,noheader,nounits)"; then
        echo "GPU0 occupancy gate failed: nvidia-smi could not query physical GPU 0" >&2
        return 1
    fi
    if [[ -n "${processes}" ]]; then
        echo "GPU0 occupancy gate failed; refusing to alter existing processes:" >&2
        echo "${processes}" >&2
        return 1
    fi
    log "GPU0 occupancy gate passed: no compute process"
}

run_stage() {
    local experiment="$1"
    local action="$2"
    check_disk
    check_gpu0

    local command=(
        "${PYTHON}"
        "${LAUNCHER}"
        --run "${experiment}"
        --action "${action}"
        --gpus 0
    )
    if [[ "${action}" == "train" ]]; then
        # The user-approved ep3 endpoint is an explicit training gate; keep
        # this flag in the command even for dry-run so the reported contract
        # is the same one that will be executed later.
        command+=(--rowbal-ep3-approved)
    fi
    if ((PREFLIGHT_ONLY == 0)); then
        command+=(--execute)
    fi
    log "+ ${command[*]}"

    if ((PREFLIGHT_ONLY == 0)); then
        "${command[@]}"
        return
    fi

    local result_file
    result_file="$(mktemp "${TMPDIR:-/tmp}/exp4_4_3_exp4_7_3_cuda0_preflight.XXXXXX")"
    if ! "${command[@]}" >"${result_file}"; then
        cat -- "${result_file}"
        rm -f -- "${result_file}"
        return 1
    fi
    cat -- "${result_file}"
    if ! "${PYTHON}" - "${result_file}" "${experiment}" "${action}" <<'PY'
import json
import sys

path, expected_experiment, expected_action = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)

if payload.get("experiment") != expected_experiment:
    raise SystemExit(
        f"preflight experiment mismatch: {payload.get('experiment')!r} "
        f"!= {expected_experiment!r}"
    )
if payload.get("action") != expected_action or payload.get("gpus") != "0":
    raise SystemExit("preflight action/GPU0 contract mismatch")
if payload.get("mode") != "dry-run" or payload.get("executed") is not False:
    raise SystemExit("preflight unexpectedly reports execution")

blockers = payload.get("blockers")
if not isinstance(blockers, list) or any(not isinstance(item, str) for item in blockers):
    raise SystemExit("preflight blocker list is malformed")
allowed = {
    "train": set(),
    "predict": {"WAIT_FINAL_SFT_TRAIN_ADAPTER"},
    "evaluate": {
        "WAIT_FINAL_SFT_TRAIN_ADAPTER",
        "WAIT_FINAL_SFT_PREDICTION",
    },
}[expected_action]
actual = set(blockers)
if not actual.issubset(allowed):
    raise SystemExit(
        f"preflight blocker set for {expected_experiment}/{expected_action} is "
        f"{sorted(actual)}, allowed={sorted(allowed)}"
    )
print(
    f"preflight accepted for {expected_experiment}/{expected_action}: "
    f"blockers={sorted(actual)}"
)
PY
    then
        rm -f -- "${result_file}"
        return 1
    fi
    rm -f -- "${result_file}"
}

# Fail-stop ordering is intentional: each completed/no-op stage is checked
# independently before the next stage consumes its artifacts.
run_stage exp4_4_3 train
run_stage exp4_4_3 predict
run_stage exp4_4_3 evaluate
run_stage exp4_7_3 train
run_stage exp4_7_3 predict
run_stage exp4_7_3 evaluate

log "PT-exp2-mm-rowbal-cont3 downstream CUDA0 batch complete"
